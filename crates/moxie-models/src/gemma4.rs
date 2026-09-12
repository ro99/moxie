//! Gemma 4 text: metadata, tensor roles and graph composition. Nothing else.
//!
//! Document 09 §B lists what a model definition may contain -- "checkpoint
//! architecture metadata interpretation and logical tensor-role mapping",
//! "graph composition and mathematical constants/options specific to that
//! family" -- and what it may not. There is no allocation here, no file read,
//! no cache, no prefill or decode loop, and no kernel. The enforcement is the
//! containing crate's three-entry dependency list, which no module can widen.
//!
//! # What this is and is not
//!
//! [`reduced`] builds a **small synthetic graph with Gemma 4's shape of
//! mathematics**, over weights the caller invents. It is a contract fixture. It
//! is not Gemma 4, it does not read the checkpoint, and its output is not model
//! output. The artifact at
//! `/fast/models/cyankiwi/gemma-4-31B-it-AWQ-8bit`, revision
//! `34ca187d836de874b2c7e3edf48f439b9f583772`, stores every language-model
//! linear as compressed-tensors INT8 `pack-quantized`; importing and executing
//! that is M3's work, and none of it is here.
//!
//! What the reduced graph does preserve, from `docs/models/gemma4.md`:
//!
//! - alternating sliding-window and full-causal layers, on the artifact's
//!   `(layer + 1) % stride == 0` global predicate;
//! - per-layer-type RoPE -- small theta and full rotation on sliding layers, a
//!   large theta and partial rotation on global layers, both half-split;
//! - grouped-query attention with a score scale of exactly 1.0;
//! - per-head query and key normalization, and a unit-gain value normalization;
//! - global layers with no value projection, taking `V` from the key
//!   projection *before* key normalization and rotation;
//! - GeGLU, four RMSNorms per layer, a per-layer scalar on the MLP residual
//!   only, an embedding output scale, tied embeddings and a logit softcap.
//!
//! What it deliberately does not preserve is in [`Reduction`].

use moxie_graph::{
    Graph, GraphBuilder, IndexEncoding, OpParams, OracleRegistry, RopeLayout, TensorSpec, ValueId,
    ValueRole, Visibility,
};
use moxie_model_api::{
    GraphRequirements, ModelDefinition, ModelMetadata, TensorRequirement, TensorRole,
};
use moxie_types::{Dim, Error, Precision, Result, SymbolId, WeightPrecision};

/// The constants the artifact's `config.json` declares, at whatever size the
/// caller asks for.
///
/// Every field is required. Gemma 4's numbers are in [`ARTIFACT`], which is
/// what the real graph would use and which no host-reference profile can run.
#[derive(Debug, Clone, PartialEq)]
pub struct TextConfig {
    /// Residual stream width. Distinct from `heads * head_dim`: the artifact's
    /// are 5376 and 8192, and a reduced graph that made them equal would stop
    /// testing the output projection's shape.
    pub hidden: u64,
    pub layers: u32,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub intermediate: u64,
    pub vocab: u64,
    /// A layer `l` is full-causal when `(l + 1) % global_stride == 0`; every
    /// other layer uses `sliding_window`. The artifact's stride is 6.
    pub global_stride: u32,
    pub sliding_window: u64,
    pub rms_eps: f32,
    pub sliding_rope_theta: f32,
    pub global_rope_theta: f32,
    /// The fraction of each head that rotates on a **global** layer. The
    /// artifact declares `partial_rotary_factor` 0.25; sliding layers rotate
    /// the whole head. The inverse-frequency denominator stays the full head
    /// dimension either way -- see [`OpParams::Rope`]'s `frequency_dim`.
    pub global_partial_rotary: Fraction,
    pub final_logit_softcap: f32,
    /// The per-layer MLP residual scalar, one per layer.
    ///
    /// In the artifact this is a loaded `layer_scalar` tensor, not a config
    /// field, so a real graph cannot be composed until it has been read. That
    /// is a genuine ordering constraint on the importer and it is recorded in
    /// the bring-up contract rather than hidden by a default here.
    pub layer_scalars: Vec<f32>,
    /// The factor applied to every looked-up embedding row. Gemma uses
    /// `bf16(sqrt(hidden))`; [`embedding_scale`] computes it.
    pub embedding_scale: f32,
    /// The longest position this configuration was built for.
    ///
    /// R19: width, admitted context and actual visible tokens are three
    /// different numbers. A reduced graph states its own small one, so the
    /// artifact's 262,144 cannot be read out of a fixture and mistaken for a
    /// supported context.
    pub max_trained_position: u64,
}

/// An exact rational fraction of a head, so a rotary fraction is never a float
/// that nearly divides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fraction {
    pub numerator: u64,
    pub denominator: u64,
}

impl Fraction {
    pub const WHOLE: Self = Self {
        numerator: 1,
        denominator: 1,
    };
    /// The artifact's `partial_rotary_factor` of 0.25.
    pub const QUARTER: Self = Self {
        numerator: 1,
        denominator: 4,
    };

    /// `width · numerator / denominator`, exactly, rounded down to an even
    /// number of elements because rotation is over pairs.
    fn of(self, width: u64) -> Result<u64> {
        if self.denominator == 0 {
            return Err(Error::InvalidRequest {
                field: "partial_rotary",
                detail: "a fraction with a zero denominator".into(),
            });
        }
        let product = width
            .checked_mul(self.numerator)
            .ok_or(Error::InvalidRequest {
                field: "partial_rotary",
                detail: "rotary width overflows".into(),
            })?;
        Ok((product / self.denominator) & !1)
    }
}

/// `bf16(sqrt(hidden))`, the embedding factor Gemma 4 applies.
///
/// Rounded to BF16 *before* it multiplies anything, because the pinned source
/// does (`src/models/gemma4/gemma4_runtime.cpp:699`): `sqrt(5376)` is
/// 73.3212... and the model multiplies by the BF16 value 73.5, not by the FP32
/// one. Computing it here keeps that rounding in the model definition, where
/// the choice belongs, rather than in an engine that would have to know why.
pub fn embedding_scale(hidden: u64) -> f32 {
    let exact = (hidden as f64).sqrt() as f32;
    // Round-to-nearest-even on the top 16 bits. The exhaustively tested
    // contract lives in `moxie-format`, which a model crate may not depend on;
    // `reduced_embedding_scale_matches_the_format_crate` in the CLI's tests
    // asserts the two agree.
    let bits = exact.to_bits();
    let lsb = (bits >> 16) & 1;
    f32::from_bits(((bits.wrapping_add(0x7FFF).wrapping_add(lsb)) >> 16) << 16)
}

/// The artifact's declared text geometry, for reference and refusal.
///
/// Nothing executes this. It is here so that a caller can compare a reduced
/// configuration against the real one, and so that the numbers in
/// `docs/models/gemma4.md` have an executable counterpart that a test can
/// check against `config.json` if the artifact is ever re-inspected.
pub const ARTIFACT: ArtifactGeometry = ArtifactGeometry {
    hidden: 5376,
    layers: 60,
    heads: 32,
    local_kv_heads: 16,
    local_head_dim: 256,
    global_kv_heads: 4,
    global_head_dim: 512,
    intermediate: 21504,
    vocab: 262_144,
    global_stride: 6,
    sliding_window: 1024,
    rms_eps: 1e-6,
    sliding_rope_theta: 10_000.0,
    global_rope_theta: 1_000_000.0,
    final_logit_softcap: 30.0,
    max_trained_position: 262_144,
};

/// The artifact's numbers. Not a [`TextConfig`]: its local and global layers
/// have different key/value head counts *and* different head dimensions, which
/// the host-reference paged profile cannot store. That difference is the whole
/// reason this is a separate type -- a `TextConfig` that could hold it would
/// imply something could run it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArtifactGeometry {
    pub hidden: u64,
    pub layers: u32,
    pub heads: u64,
    pub local_kv_heads: u64,
    pub local_head_dim: u64,
    pub global_kv_heads: u64,
    pub global_head_dim: u64,
    pub intermediate: u64,
    pub vocab: u64,
    pub global_stride: u32,
    pub sliding_window: u64,
    pub rms_eps: f32,
    pub sliding_rope_theta: f32,
    pub global_rope_theta: f32,
    pub final_logit_softcap: f32,
    pub max_trained_position: u64,
}

impl ArtifactGeometry {
    /// Whether layer `l` uses full causal attention.
    pub const fn global_layer(&self, layer: u32) -> bool {
        layer < self.layers && (layer + 1).is_multiple_of(self.global_stride)
    }
}

/// What a reduced graph gives up, enumerated rather than described.
///
/// Returned by [`Gemma4Text::reduction`] so that a diagnostic surface can print
/// it. A reduced graph that could not say what it reduced would be exactly the
/// "synthetic output described as model support" that document 06 forbids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reduction {
    /// Uniform key/value geometry across layers. The artifact's sliding layers
    /// are 16 heads of 256 and its global layers 4 of 512. Per-layer paged
    /// geometry is M4.
    pub uniform_kv_geometry: bool,
    /// Sliding layers retain their whole history rather than a window-sized
    /// ring. The mask is exact; the memory saving is M4.
    pub sliding_layers_retain_full_history: bool,
    /// Weights are whatever the caller supplied. The artifact is INT8
    /// `pack-quantized` and its importer is M3.
    pub synthetic_weights: bool,
    /// No vision tower, no image tokens, no multimodal mask exemption. M11.
    pub text_only: bool,
}

impl Reduction {
    /// Everything [`Gemma4Text::reduced`] gives up. Public because the surface
    /// that prints the disclosure needs it before a graph exists.
    pub const fn all() -> Self {
        Self::ALL
    }

    const ALL: Self = Self {
        uniform_kv_geometry: true,
        sliding_layers_retain_full_history: true,
        synthetic_weights: true,
        text_only: true,
    };
}

/// A logical tensor role and the graph value that carries it.
///
/// The composition root uses this to bind weights: the model names roles, the
/// caller supplies bytes, and neither knows the other's file names. Shapes come
/// from `graph.spec(value)`, so there is one description of a tensor's extent
/// and it is the graph's.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundRole {
    pub role: TensorRole,
    pub value: ValueId,
}

/// A composed graph and everything a caller needs to fill it.
#[derive(Debug)]
pub struct Composition {
    pub graph: Graph,
    pub tokens: ValueId,
    pub positions: ValueId,
    pub weights: Vec<BoundRole>,
}

/// Gemma 4's text tower as a model definition.
#[derive(Debug, Clone)]
pub struct Gemma4Text {
    config: TextConfig,
    metadata: ModelMetadata,
    tensors: Vec<TensorRequirement>,
}

fn role(name: &str, layer: Option<u32>) -> TensorRole {
    TensorRole {
        name: name.to_string(),
        layer,
        expert: None,
    }
}

fn required(name: &str, layer: Option<u32>) -> Result<TensorRequirement> {
    Ok(TensorRequirement {
        role: role(name, layer),
        // BF16 only, and that is a statement about this graph rather than about
        // the family: the artifact's linears are INT8 and would need M3's
        // importer plus a second allowed precision here. Listing INT8 now would
        // advertise a path that does not exist.
        allowed: vec![WeightPrecision::new(Precision::Bf16)?],
        required: true,
    })
}

impl Gemma4Text {
    /// A reduced Gemma-4-like text tower.
    ///
    /// `revision` names what the caller is modelling; it is recorded in the
    /// metadata and never interpreted. Passing the artifact's revision here
    /// does not make this the artifact, and the reduction is reported
    /// alongside it.
    pub fn reduced(config: TextConfig, revision: &str) -> Result<Self> {
        config.check()?;
        let metadata = ModelMetadata {
            family: "gemma4-text-reduced".into(),
            revision: revision.into(),
            max_trained_position: config.max_trained_position,
            vocab_size: u32::try_from(config.vocab).map_err(|_| Error::InvalidArtifact {
                detail: format!("vocabulary {} does not fit a u32", config.vocab),
            })?,
        };
        let mut tensors = vec![required("embedding", None)?, required("final_norm", None)?];
        for layer in 0..config.layers {
            for name in [
                "attn_norm",
                "q_proj",
                "k_proj",
                "q_norm",
                "k_norm",
                "v_norm_unit_gain",
                "o_proj",
                "attn_out_norm",
                "ffn_norm",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
                "ffn_out_norm",
            ] {
                tensors.push(required(name, Some(layer))?);
            }
            // Exactly the artifact's structure: the ten global layers have no
            // `v_proj`, because `attention_k_eq_v` makes V the key projection.
            if !config.global_layer(layer) {
                tensors.push(required("v_proj", Some(layer))?);
            }
        }
        Ok(Self {
            config,
            metadata,
            tensors,
        })
    }

    pub fn config(&self) -> &TextConfig {
        &self.config
    }

    /// What this graph is not. See [`Reduction`].
    pub const fn reduction(&self) -> Reduction {
        Reduction::ALL
    }

    /// Whether layer `l` uses full causal attention rather than a window.
    pub const fn global_layer(&self, layer: u32) -> bool {
        self.config.global_layer(layer)
    }

    /// Compose the graph.
    ///
    /// The builder is the shared one and every node is a shared operation with
    /// a registered oracle; `finish` refuses otherwise. There is no branch here
    /// on anything but the layer type, and no arithmetic that is not an
    /// `OpParams` field.
    pub fn compose(&self, oracles: &OracleRegistry, rows: SymbolId) -> Result<Composition> {
        let c = &self.config;
        // Checked, and checked *here*: these products become tensor extents and
        // shape arguments, and `GraphBuilder` never sees them as a
        // multiplication it could check for us. An independent review wrapped
        // `heads * head_dim` with `heads = 2^63` and reached a panic through a
        // public entry point, which is neither the checked arithmetic nor the
        // typed error this repository requires.
        let query_width = width("heads * head_dim", c.heads, c.head_dim)?;
        let kv_width = width("kv_heads * head_dim", c.kv_heads, c.head_dim)?;
        let mut g = GraphBuilder::new(moxie_graph::OracleId("moxie_oracles::host_reference"), rows);
        let index = TensorSpec::new(
            ValueRole::Index(IndexEncoding::U64),
            vec![Dim::symbol(rows)],
        );
        let tokens = g.input("tokens", index.clone());
        let positions = g.input("absolute positions", index);

        let mut bound = Vec::new();
        let weight = |g: &mut GraphBuilder,
                      bound: &mut Vec<BoundRole>,
                      name: &str,
                      layer: Option<u32>,
                      shape: Vec<u64>|
         -> Result<ValueId> {
            let label = match layer {
                Some(l) => format!("{name}.{l}"),
                None => name.to_string(),
            };
            let id = g.weight(
                &label,
                TensorSpec::new(
                    ValueRole::Weight(WeightPrecision::new(Precision::Bf16)?),
                    shape.into_iter().map(Dim::constant).collect(),
                ),
            )?;
            bound.push(BoundRole {
                role: role(name, layer),
                value: id,
            });
            Ok(id)
        };

        // Tied: the same table embeds and projects. The artifact has no
        // `lm_head` tensor, so a graph with two of them would be describing a
        // different checkpoint.
        // The remaining extents are products too. `vocab x hidden` is the
        // largest tensor in any real configuration, so it is the one most
        // likely to overflow a caller's arithmetic before it reaches a shape.
        width("vocab * hidden", c.vocab, c.hidden)?;
        width("intermediate * hidden", c.intermediate, c.hidden)?;
        let embedding = weight(
            &mut g,
            &mut bound,
            "embedding",
            None,
            vec![c.vocab, c.hidden],
        )?;
        let final_norm = weight(&mut g, &mut bound, "final_norm", None, vec![c.hidden])?;

        let mut stream = g.node(
            OpParams::Embedding {
                vocab: c.vocab,
                hidden: c.hidden,
                scale: c.embedding_scale,
            },
            &[tokens, embedding],
        )?;

        for layer in 0..c.layers {
            let global = c.global_layer(layer);
            let attn_norm = weight(&mut g, &mut bound, "attn_norm", Some(layer), vec![c.hidden])?;
            let normed = g.node(
                OpParams::RmsNorm {
                    hidden: c.hidden,
                    group: 1,
                    eps: c.rms_eps,
                },
                &[stream, attn_norm],
            )?;

            let wq = weight(
                &mut g,
                &mut bound,
                "q_proj",
                Some(layer),
                vec![query_width, c.hidden],
            )?;
            let wk = weight(
                &mut g,
                &mut bound,
                "k_proj",
                Some(layer),
                vec![kv_width, c.hidden],
            )?;
            let q = linear(&mut g, normed, wq, c.hidden, query_width)?;
            let k = linear(&mut g, normed, wk, c.hidden, kv_width)?;
            // `attention_k_eq_v`: a global layer's value stream is the key
            // projection's output, taken here -- before key normalization and
            // rotation -- exactly as the pinned runtime copies it
            // (`gemma4_runtime.cpp:779`). Taking it after would make V a
            // rotated tensor, which it is not.
            let v = if global {
                k
            } else {
                let wv = weight(
                    &mut g,
                    &mut bound,
                    "v_proj",
                    Some(layer),
                    vec![kv_width, c.hidden],
                )?;
                linear(&mut g, normed, wv, c.hidden, kv_width)?
            };

            let q_norm = weight(&mut g, &mut bound, "q_norm", Some(layer), vec![c.head_dim])?;
            let k_norm = weight(&mut g, &mut bound, "k_norm", Some(layer), vec![c.head_dim])?;
            // The value normalization's gain is all ones in the pinned source,
            // which passes a literal vector of them. It is a real weight here
            // rather than an implicit special case, so that "normalize V with
            // a unit gain" is visible in the graph and bindable by a caller
            // that has a different one.
            let v_gain = weight(
                &mut g,
                &mut bound,
                "v_norm_unit_gain",
                Some(layer),
                vec![c.head_dim],
            )?;
            let qn = g.node(
                OpParams::RmsNorm {
                    hidden: query_width,
                    group: c.heads,
                    eps: c.rms_eps,
                },
                &[q, q_norm],
            )?;
            let kn = g.node(
                OpParams::RmsNorm {
                    hidden: kv_width,
                    group: c.kv_heads,
                    eps: c.rms_eps,
                },
                &[k, k_norm],
            )?;
            let vn = g.node(
                OpParams::RmsNorm {
                    hidden: kv_width,
                    group: c.kv_heads,
                    eps: c.rms_eps,
                },
                &[v, v_gain],
            )?;

            let (theta, rotary) = if global {
                (c.global_rope_theta, c.global_partial_rotary.of(c.head_dim)?)
            } else {
                (c.sliding_rope_theta, c.head_dim)
            };
            let qr = g.node(
                OpParams::Rope {
                    heads: c.heads,
                    head_dim: c.head_dim,
                    rotary_dim: rotary,
                    // The full head dimension, not the rotated width: a global
                    // layer rotates a quarter of the head and still divides by
                    // the whole of it.
                    frequency_dim: c.head_dim,
                    base: theta,
                    layout: RopeLayout::HalfSplit,
                },
                &[qn, positions],
            )?;
            let kr = g.node(
                OpParams::Rope {
                    heads: c.kv_heads,
                    head_dim: c.head_dim,
                    rotary_dim: rotary,
                    frequency_dim: c.head_dim,
                    base: theta,
                    layout: RopeLayout::HalfSplit,
                },
                &[kn, positions],
            )?;

            let attention = g.node(
                OpParams::Attention {
                    heads: c.heads,
                    kv_heads: c.kv_heads,
                    head_dim: c.head_dim,
                    // Exactly 1.0. The queries and keys were just normalized;
                    // the pinned runtime passes `scale = 1.0F` and dividing by
                    // the square root of the head dimension here would be a
                    // constant error inside every softmax.
                    scale: 1.0,
                    visibility: if global {
                        Visibility::Causal
                    } else {
                        Visibility::SlidingWindow {
                            window: c.sliding_window,
                        }
                    },
                    layer,
                },
                &[qr, kr, vn, positions],
            )?;

            let wo = weight(
                &mut g,
                &mut bound,
                "o_proj",
                Some(layer),
                vec![c.hidden, query_width],
            )?;
            let projected = linear(&mut g, attention, wo, query_width, c.hidden)?;
            let attn_out_norm = weight(
                &mut g,
                &mut bound,
                "attn_out_norm",
                Some(layer),
                vec![c.hidden],
            )?;
            let normed_out = g.node(
                OpParams::RmsNorm {
                    hidden: c.hidden,
                    group: 1,
                    eps: c.rms_eps,
                },
                &[projected, attn_out_norm],
            )?;
            // Unscaled. The layer scalar belongs to the MLP residual only.
            stream = g.node(OpParams::Residual { scale: 1.0 }, &[stream, normed_out])?;

            let ffn_norm = weight(&mut g, &mut bound, "ffn_norm", Some(layer), vec![c.hidden])?;
            let pre = g.node(
                OpParams::RmsNorm {
                    hidden: c.hidden,
                    group: 1,
                    eps: c.rms_eps,
                },
                &[stream, ffn_norm],
            )?;
            let w_gate = weight(
                &mut g,
                &mut bound,
                "ffn_gate",
                Some(layer),
                vec![c.intermediate, c.hidden],
            )?;
            let w_up = weight(
                &mut g,
                &mut bound,
                "ffn_up",
                Some(layer),
                vec![c.intermediate, c.hidden],
            )?;
            let w_down = weight(
                &mut g,
                &mut bound,
                "ffn_down",
                Some(layer),
                vec![c.hidden, c.intermediate],
            )?;
            let gate = linear(&mut g, pre, w_gate, c.hidden, c.intermediate)?;
            let up = linear(&mut g, pre, w_up, c.hidden, c.intermediate)?;
            let activated = g.node(
                OpParams::GeGlu {
                    width: c.intermediate,
                },
                &[gate, up],
            )?;
            let down = linear(&mut g, activated, w_down, c.intermediate, c.hidden)?;
            let ffn_out_norm = weight(
                &mut g,
                &mut bound,
                "ffn_out_norm",
                Some(layer),
                vec![c.hidden],
            )?;
            let normed_ffn = g.node(
                OpParams::RmsNorm {
                    hidden: c.hidden,
                    group: 1,
                    eps: c.rms_eps,
                },
                &[down, ffn_out_norm],
            )?;
            stream = g.node(
                OpParams::Residual {
                    scale: c.layer_scalars[layer as usize],
                },
                &[stream, normed_ffn],
            )?;
        }

        let normed = g.node(
            OpParams::RmsNorm {
                hidden: c.hidden,
                group: 1,
                eps: c.rms_eps,
            },
            &[stream, final_norm],
        )?;
        let logits = g.node(
            OpParams::VocabProjection {
                vocab: c.vocab,
                hidden: c.hidden,
                softcap: Some(c.final_logit_softcap),
            },
            &[normed, embedding],
        )?;

        Ok(Composition {
            graph: g.finish(logits, oracles)?,
            tokens,
            positions,
            weights: bound,
        })
    }
}

/// A head-count times head-dimension product, as a checked tensor extent.
fn width(what: &'static str, heads: u64, head_dim: u64) -> Result<u64> {
    heads
        .checked_mul(head_dim)
        .filter(|w| *w <= u64::from(u32::MAX))
        .ok_or(Error::InvalidRequest {
            field: "attention",
            detail: format!("{what} = {heads} x {head_dim} is not a representable tensor extent"),
        })
}

fn linear(
    g: &mut GraphBuilder,
    x: ValueId,
    w: ValueId,
    in_features: u64,
    out_features: u64,
) -> Result<ValueId> {
    g.node(
        OpParams::Linear {
            in_features,
            out_features,
            // Gemma 4 declares `attention_bias: false` and has no bias tensor
            // in its index. An unconditional bias here would require inventing
            // one.
            bias: false,
        },
        &[x, w],
    )
}

impl TextConfig {
    const fn global_layer(&self, layer: u32) -> bool {
        self.global_stride != 0
            && layer < self.layers
            && (layer + 1).is_multiple_of(self.global_stride)
    }

    fn check(&self) -> Result<()> {
        let bad = |field: &'static str, detail: String| Error::InvalidRequest { field, detail };
        if self.layers == 0 || self.layer_scalars.len() != self.layers as usize {
            return Err(bad(
                "layer_scalars",
                format!(
                    "{} scalar(s) for {} layer(s); the MLP residual factor is per layer",
                    self.layer_scalars.len(),
                    self.layers
                ),
            ));
        }
        if self.max_trained_position == 0 {
            return Err(bad(
                "max_trained_position",
                "a graph trained for no positions".into(),
            ));
        }
        if self.global_stride == 0 {
            return Err(bad(
                "global_stride",
                "a stride of zero makes every layer sliding and never rotates the \
                 global theta; state the stride the checkpoint declares"
                    .into(),
            ));
        }
        if self.sliding_window == 0 {
            return Err(bad(
                "sliding_window",
                "a window of zero would let a sliding layer see nothing, not everything".into(),
            ));
        }
        for (i, s) in self.layer_scalars.iter().enumerate() {
            if !(s.is_finite() && *s > 0.0) {
                return Err(bad(
                    "layer_scalars",
                    format!("layer {i} scalar {s} is not finite and positive"),
                ));
            }
        }
        // The remaining bounds -- head divisibility, rotary width, epsilon,
        // theta, the softcap -- are `OpParams::check_params`' job and are
        // checked when the node is built. Repeating them here would be a
        // second contract that could drift from the first.
        Ok(())
    }
}

impl ModelDefinition for Gemma4Text {
    fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    fn tensors(&self) -> &[TensorRequirement] {
        &self.tensors
    }

    fn graph_requirements(&self) -> GraphRequirements {
        use moxie_graph::Op;
        GraphRequirements {
            ops: vec![
                Op::Embedding,
                Op::RmsNorm,
                Op::Linear,
                Op::Rope,
                Op::Attention,
                Op::Residual,
                Op::GeGlu,
                Op::VocabProjection,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use moxie_graph::Op;

    /// A configuration small enough for the host-reference profile.
    fn reduced_config() -> TextConfig {
        TextConfig {
            hidden: 24,
            layers: 6,
            heads: 4,
            kv_heads: 2,
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
        }
    }

    #[test]
    fn the_global_predicate_matches_the_artifact() {
        // `docs/models/gemma4.md`: the ten layers without a `v_proj` are
        // 5, 11, 17, 23, 29, 35, 41, 47, 53, 59.
        let global: Vec<u32> = (0..60).filter(|l| ARTIFACT.global_layer(*l)).collect();
        assert_eq!(global, vec![5, 11, 17, 23, 29, 35, 41, 47, 53, 59]);
        assert_eq!(global.len(), 10);
    }

    #[test]
    fn a_global_layer_declares_no_value_projection() {
        let m = Gemma4Text::reduced(reduced_config(), "synthetic").unwrap();
        let v_layers: Vec<Option<u32>> = m
            .tensors()
            .iter()
            .filter(|t| t.role.name == "v_proj")
            .map(|t| t.role.layer)
            .collect();
        // Six layers, stride six: layer 5 is global and has no v_proj.
        assert_eq!(
            v_layers,
            vec![Some(0), Some(1), Some(2), Some(3), Some(4)],
            "the global layer must not require a value projection"
        );
        assert!(m.global_layer(5) && !m.global_layer(4));
    }

    #[test]
    fn the_partial_rotary_fraction_is_exact_and_even() {
        assert_eq!(Fraction::QUARTER.of(512).unwrap(), 128);
        assert_eq!(Fraction::WHOLE.of(256).unwrap(), 256);
        // Rotation is over pairs, so an odd result is rounded down rather than
        // accepted and then rejected by the node.
        assert_eq!(Fraction::QUARTER.of(8).unwrap(), 2);
        assert_eq!(
            Fraction {
                numerator: 1,
                denominator: 3
            }
            .of(8)
            .unwrap(),
            2
        );
    }

    #[test]
    fn a_layer_scalar_per_layer_is_required() {
        let mut c = reduced_config();
        c.layer_scalars.pop();
        assert!(Gemma4Text::reduced(c.clone(), "x").is_err());
        c.layer_scalars = vec![1.0; 6];
        assert!(Gemma4Text::reduced(c.clone(), "x").is_ok());
        c.layer_scalars[2] = 0.0;
        assert!(Gemma4Text::reduced(c, "x").is_err());
    }

    #[test]
    fn the_artifact_geometry_cannot_be_a_text_config() {
        // Not a compile-time claim: the point is that the artifact's local and
        // global layers differ in both key/value head count and head
        // dimension, so no single uniform configuration describes it.
        assert_ne!(ARTIFACT.local_kv_heads, ARTIFACT.global_kv_heads);
        assert_ne!(ARTIFACT.local_head_dim, ARTIFACT.global_head_dim);
    }

    #[test]
    fn an_overflowing_extent_is_a_typed_error_not_a_panic() {
        // Independent review finding, reproduced. `TextConfig::check` delegates
        // dimension validation to graph construction, so the products that
        // precede construction have to carry their own checks.
        let mut oracles = OracleRegistry::new();
        for op in [
            Op::Embedding,
            Op::Linear,
            Op::RmsNorm,
            Op::Rope,
            Op::Attention,
            Op::Residual,
            Op::GeGlu,
            Op::VocabProjection,
        ] {
            oracles
                .register(
                    op,
                    moxie_graph::OracleId("moxie_oracles::host_reference"),
                    moxie_graph::OracleEvidence {
                        implementation: "test",
                        test_module: "test",
                    },
                )
                .unwrap();
        }
        // Each case names the guard it must reach, so a configuration refused
        // by an *earlier* guard fails the test instead of quietly passing it.
        // Re-review found exactly that: the `kv_heads` case also overflowed
        // `heads`, so the query-width guard rejected it first and the
        // key/value guard was never exercised. `TextConfig::check` does not
        // require `kv_heads <= heads` -- that belongs to
        // `OpParams::check_params`, which runs after these products are formed
        // -- so a lone oversized `kv_heads` really does reach the
        // multiplication.
        for (field, mutate, expect) in [
            (
                "heads",
                (|c: &mut TextConfig| c.heads = 1 << 63) as fn(&mut TextConfig),
                "heads * head_dim",
            ),
            (
                "kv_heads",
                |c: &mut TextConfig| c.kv_heads = 1 << 60,
                "kv_heads * head_dim",
            ),
            (
                "head_dim",
                |c: &mut TextConfig| c.head_dim = 1 << 60,
                "heads * head_dim",
            ),
            (
                "vocab",
                |c: &mut TextConfig| c.vocab = u64::MAX,
                "does not fit a u32",
            ),
            (
                "intermediate",
                |c: &mut TextConfig| c.intermediate = u64::MAX,
                "intermediate * hidden",
            ),
        ] {
            let mut config = reduced_config();
            mutate(&mut config);
            // Refusal at either stage is correct -- `vocab` is already rejected
            // by the metadata's `u32` conversion. What must not happen is a
            // wrapped extent or a panic.
            let error = match Gemma4Text::reduced(config, "overflow") {
                Err(e) => e,
                Ok(model) => model
                    .compose(&oracles, SymbolId(0))
                    .expect_err("construction must refuse rather than wrap or panic"),
            };
            let detail = format!("{error}");
            assert!(
                detail.contains(expect),
                "{field}: refused by the wrong guard -- wanted {expect:?}, got {detail:?}"
            );
        }
    }

    #[test]
    fn the_embedding_scale_is_rounded_before_it_multiplies() {
        // sqrt(5376) = 73.3212..., whose BF16 neighbour is 73.5.
        let s = embedding_scale(5376);
        assert_eq!(s, 73.5, "got {s}");
        assert_ne!(s, (5376f64).sqrt() as f32);
        // A perfect square still lands on itself.
        assert_eq!(embedding_scale(64), 8.0);
    }
}
