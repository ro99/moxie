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
use moxie_models::gemma4::{
    Fraction, Gemma4Text, MoeGeometry, Reduction, TextConfig, embedding_scale, router_input_scale,
};
use moxie_types::{Dim, Error, Result, SymbolId};

/// Which reduced geometry to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Six layers on the artifact's own 1-in-6 global stride, so five sliding
    /// layers precede one full-causal layer. 4 query heads, over 2 key/value
    /// heads of 16 when sliding and 1 of 32 when global -- the artifact's own
    /// asymmetry (16x256 sliding, 4x512 global) at a size that runs.
    A,
    /// Three layers on a 1-in-3 stride, with different head **ratios**, head
    /// dimensions, window and residual-stream width. Shape A narrows its
    /// key/value heads on a global layer and widens the head; this one widens
    /// the heads instead, so neither direction can be load-bearing by accident.
    B,
    /// Shape A's attention geometry with the designated M2 artifact's routed
    /// block: a dense shared expert **and** routed experts in every layer, the
    /// router reading the un-normalized residual, and the three extra norms.
    ///
    /// Deliberately a third shape rather than a flag on A. A and B stay dense
    /// and bit-for-bit what tasks 0016 and 0017 gated, so a routing defect
    /// cannot hide behind a changed baseline.
    C,
}

impl Shape {
    pub fn config(self) -> TextConfig {
        match self {
            Shape::A => TextConfig {
                hidden: 24,
                layers: 6,
                heads: 4,
                local_kv_heads: 2,
                // Sixteen, not eight. A quarter of an eight-wide head is a
                // single rotated pair, whose inverse frequency is `base^0` --
                // so the global theta would cancel and the geometry would
                // silently stop testing it. Sixteen gives the global layers two
                // angles, which is the smallest size at which the base matters.
                local_head_dim: 16,
                // A different head count *and* a different head dimension on
                // the global layer, so the query width, the key/value width and
                // the paged row width all change with the layer type.
                global_kv_heads: 1,
                global_head_dim: 32,
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
                // Dense, like the 31B this shape was built for. Shape C carries
                // the routed block; keeping A and B dense is what lets task
                // 0016's and 0017's gates stay bit-for-bit comparable.
                moe: None,
            },
            Shape::B => TextConfig {
                hidden: 12,
                layers: 3,
                heads: 6,
                // Three query heads per key/value head, against shape A's two.
                local_kv_heads: 2,
                // Half of eight is four: two angles again, for the same reason.
                local_head_dim: 8,
                // Two query heads per key/value head here, and a wider head:
                // the opposite adjustment to shape A's global layer.
                global_kv_heads: 3,
                global_head_dim: 12,
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
                moe: None,
            },
            Shape::C => TextConfig {
                moe: Some(MoeGeometry {
                    // Five experts choosing two: small enough to run, and wide
                    // enough that the union of a multi-row batch's routes is
                    // strictly smaller than `rows * top_k`, which is the
                    // property document 03 warns against losing.
                    experts: 5,
                    top_k: 2,
                    // Narrower than the dense `intermediate` of 16, as the
                    // artifact's 704 is narrower than its 2,112. Equal widths
                    // would let a graph that confused the two branches pass.
                    moe_intermediate: 6,
                    router_input_scale: router_input_scale(24),
                }),
                ..Shape::A.config()
            },
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Shape::A => "gemma-a",
            Shape::B => "gemma-b",
            Shape::C => "gemma-c-moe",
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
