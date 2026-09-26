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

use std::collections::BTreeMap;
use std::path::Path;

use crate::fixture::Fixture;
use moxie_engine::{HostTensor, Value};
use moxie_format::checkpoint_config;
use moxie_format::safetensors::Dtype;
use moxie_graph::{Bindings, OracleRegistry};
use moxie_model_api::{ModelDefinition, TensorRole};
use moxie_models::gemma4::{
    Composition, Fraction, Gemma4Text, MoeGeometry, Reduction, TextConfig, embedding_scale,
    gemma4_source_tensor, router_input_scale, text_config_from_declared,
};
use moxie_storage::{Shard, read_text_capped};
use moxie_types::{
    Dim, Error, Precision, Result, SymbolId, WeightPrecision, declared_value::DeclaredValue,
};

/// A deterministic tensor role key, including the fields that distinguish
/// routed or future expert roles without requiring an ordering on TensorRole.
pub type TensorRoleKey = (String, Option<u32>, Option<u32>);

/// A full-size checkpoint graph and its source identities.
#[derive(Debug)]
pub struct CheckpointGraph {
    pub composition: Composition,
    pub role_to_source_tensor: BTreeMap<TensorRoleKey, Option<String>>,
    pub revision: String,
    pub config: TextConfig,
}

/// Build a full-size Gemma 4 text graph from a downloaded checkpoint
/// directory (ADR 0038: run source checkpoints directly).
///
/// Reads `config.json`, the safetensors index and every `layer_scalar`
/// tensor, and composes the graph with the checkpoint's own declared
/// precisions. It loads no other weight -- loading is route item 5.
pub fn from_checkpoint(dir: &Path) -> Result<CheckpointGraph> {
    let config_text = read_text_capped(
        &dir.join("config.json"),
        checkpoint_config::MAX_CONFIG_BYTES,
    )?;
    let declaration = checkpoint_config::parse(&config_text)?;
    if let Some(quantization) = &declaration.quantization
        && let Some(pattern) = quantization.passthrough_patterns.first()
    {
        // An AutoRound 16-bit passthrough override marks a module as
        // unquantized source, which the ignore-list check below does not
        // see. Composing it INT8 anyway would be a wrong graph, not a
        // refusal, so this is refused rather than matched: there is no
        // consumer of the pattern yet.
        return Err(Error::InvalidArtifact {
            detail: format!(
                "this checkpoint declares 16-bit passthrough override {pattern:?}; from_checkpoint \
                 does not yet match passthrough patterns against composed roles"
            )
            .into(),
        });
    }
    let fields = checkpoint_config::declared_text_fields(&config_text)?;

    let index_text = read_text_capped(
        &dir.join("model.safetensors.index.json"),
        checkpoint_config::MAX_INDEX_BYTES,
    )?;
    let weight_map = checkpoint_config::parse_index(&index_text)?;

    let layers = match fields.get("num_hidden_layers") {
        Some(DeclaredValue::Int(n)) if *n > 0 => {
            u32::try_from(*n).map_err(|_| Error::InvalidArtifact {
                detail: format!("num_hidden_layers {n} does not fit a u32").into(),
            })?
        }
        other => {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "num_hidden_layers is {other:?}; expected a positive declared integer"
                )
                .into(),
            });
        }
    };

    let mut shards: BTreeMap<String, Shard> = BTreeMap::new();
    let mut layer_scalars = Vec::with_capacity(layers as usize);
    for layer in 0..layers {
        let name = format!("model.language_model.layers.{layer}.layer_scalar");
        let file = weight_map
            .get(&name)
            .ok_or_else(|| Error::InvalidArtifact {
                detail: format!("the index does not declare {name}").into(),
            })?;
        if !shards.contains_key(file) {
            shards.insert(file.clone(), Shard::open(&dir.join(file))?);
        }
        let shard = &shards[file];
        let entry = shard.header().get(&name)?;
        if entry.dtype != Dtype::Bf16 || entry.shape != [1] {
            return Err(Error::InvalidArtifact {
                detail: format!(
                    "{name} is {:?} with shape {:?}; expected one BF16 scalar",
                    entry.dtype, entry.shape
                )
                .into(),
            });
        }
        let bytes = shard.tensor_bytes(&name)?;
        let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        layer_scalars.push(moxie_format::bf16::bf16_bits_to_f32(bits));
    }

    let config = text_config_from_declared(&fields, layer_scalars)?;
    let revision = checkpoint_revision(dir)?;
    let model = Gemma4Text::full(config.clone(), &revision)?;

    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles)?;
    moxie_model_api::admit(&model, &oracles)?;

    // The precision each linear composes at: the checkpoint's declared
    // integer width when it is quantized and not in the `ignore` list, BF16
    // otherwise -- `weight()` already defaults every value to BF16, so only
    // the deviations need an entry.
    let mut precisions = BTreeMap::new();
    let mut role_to_source_tensor = BTreeMap::new();
    for tensor in model.tensors() {
        let source = gemma4_source_tensor(&tensor.role);
        let key = (
            tensor.role.name.clone(),
            tensor.role.layer,
            tensor.role.expert,
        );
        role_to_source_tensor.insert(key, source.clone());

        let Some(source) = &source else { continue };
        let is_linear = tensor
            .allowed
            .iter()
            .any(|precision| precision.get() == Precision::Int8);
        if !is_linear {
            continue;
        }
        let Some(quantization) = &declaration.quantization else {
            continue;
        };
        if quantization.ignored.iter().any(|module| module == source) {
            continue;
        }
        let quantized = match quantization.bits {
            8 => Precision::Int8,
            4 => Precision::Int4,
            bits => {
                return Err(Error::InvalidArtifact {
                    detail: format!(
                        "{source} declares {bits}-bit weights; only 4 and 8 compose here"
                    )
                    .into(),
                });
            }
        };
        precisions.insert(weight_label(&tensor.role), WeightPrecision::new(quantized)?);
    }

    let composition = model.compose_with_weight_precisions(&oracles, SymbolId(0), precisions)?;

    Ok(CheckpointGraph {
        composition,
        role_to_source_tensor,
        revision,
        config,
    })
}

/// The name a composed graph gives a role's weight value. Mirrors
/// `Gemma4Text`'s own private `weight()` helper -- there is one labelling
/// rule, and this is the composition root's copy of it, needed to key the
/// precision map `compose_with_weight_precisions` reads by name.
fn weight_label(role: &TensorRole) -> String {
    match role.layer {
        Some(layer) => format!("{}.{layer}", role.name),
        None => role.name.clone(),
    }
}

/// The checkpoint's own recorded revision: the first line of the local Hub
/// cache metadata beside `config.json`, which the hub's own download
/// metadata format always writes as a 40-character lowercase hex commit id.
/// There is no fallback -- a label that is not the checkpoint's actual
/// revision is worse than a refusal naming the missing file.
fn checkpoint_revision(dir: &Path) -> Result<String> {
    let metadata_path = dir.join(".cache/huggingface/download/config.json.metadata");
    let text = std::fs::read_to_string(&metadata_path).map_err(|e| Error::InvalidArtifact {
        detail: format!("{}: {e}", metadata_path.display()).into(),
    })?;
    let revision = text.lines().next().unwrap_or("").trim();
    let is_revision_hex = revision.len() == 40
        && revision
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if !is_revision_hex {
        return Err(Error::InvalidArtifact {
            detail: format!(
                "{}: first line {revision:?} is not a 40-character lowercase hex revision",
                metadata_path.display()
            )
            .into(),
        });
    }
    Ok(revision.to_string())
}

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
                detail: format!("role {} has no graph value", bound.role.name).into(),
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
                    )
                    .into(),
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
