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
    CombineOrder, ExpertActivation, Graph, GraphBuilder, IndexEncoding, OpParams, OracleRegistry,
    RopeLayout, TensorSpec, ValueId, ValueRole, Visibility,
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
    /// Key/value heads and head dimension on a **sliding** layer.
    ///
    /// Separate from the global pair because the artifact's are not the same:
    /// 16 heads of 256 on its sliding layers, 4 of 512 on its global ones. One
    /// pair for both would describe a model that does not exist, and the query
    /// width `heads * head_dim` differs between the two layer types with it.
    pub local_kv_heads: u64,
    pub local_head_dim: u64,
    /// Key/value heads and head dimension on a **global** layer.
    pub global_kv_heads: u64,
    pub global_head_dim: u64,
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
    /// The routed-expert block, when the variant has one.
    ///
    /// `None` is the dense variant: `enable_moe_block` false, as the 31B
    /// declares. `Some` adds routed experts **beside** the dense MLP, never
    /// instead of it -- every layer of the routed variant carries both, and the
    /// dense one is a shared expert outside routing.
    pub moe: Option<MoeGeometry>,
}

/// A routed-expert block's declared geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeGeometry {
    pub experts: u64,
    pub top_k: u64,
    /// One expert's intermediate width.
    ///
    /// Distinct from [`TextConfig::intermediate`], and in the designated
    /// artifact much smaller: 704 against the dense MLP's 2,112. A layer
    /// carries both, so one field could not describe it.
    pub moe_intermediate: u64,
    /// The scalar the router applies to its normalized, gained input.
    ///
    /// [`router_input_scale`] computes the family's value. Stored rather than
    /// derived at composition time for the same reason
    /// [`TextConfig::embedding_scale`] is: it is a family choice, and deriving
    /// it inside a shared operation would impose Gemma's on everything else.
    pub router_input_scale: f32,
}

/// `hidden^(-1/2)`, the factor Gemma 4's router applies to its input.
///
/// `Gemma4TextRouter.scalar_root_size` in the pinned `transformers` source.
/// Computed in FP64 and narrowed once, and **not** pre-rounded to BF16: unlike
/// the embedding scale, the pinned source keeps it a Python float and lets the
/// multiplication carry it, so rounding it here would be a boundary the
/// reference does not have.
pub fn router_input_scale(hidden: u64) -> f32 {
    (hidden as f64).sqrt().recip() as f32
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
    moe: None,
};

/// The designated M2 artifact's declared text geometry.
///
/// `/fast/models/google/gemma-4-26B-A4B-it`, revision
/// `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, read from its own `config.json`
/// and safetensors headers on 2026-09-12. The same family as [`ARTIFACT`] --
/// same global predicate, same layer-type asymmetry, same softcap, same window
/// -- with a routed-expert block the 31B does not have.
///
/// Nothing executes this either. It is here so a caller can compare a reduced
/// configuration against the real one, and so that the numbers in
/// `docs/models/gemma4.md` have an executable counterpart.
pub const ARTIFACT_A4B: ArtifactGeometry = ArtifactGeometry {
    hidden: 2816,
    layers: 30,
    heads: 16,
    local_kv_heads: 8,
    local_head_dim: 256,
    global_kv_heads: 2,
    global_head_dim: 512,
    intermediate: 2112,
    vocab: 262_144,
    global_stride: 6,
    sliding_window: 1024,
    rms_eps: 1e-6,
    sliding_rope_theta: 10_000.0,
    global_rope_theta: 1_000_000.0,
    final_logit_softcap: 30.0,
    max_trained_position: 262_144,
    moe: Some(MoeGeometry {
        experts: 128,
        top_k: 8,
        moe_intermediate: 704,
        // `2816^(-1/2)`. A `const` cannot call `router_input_scale`, so
        // `the_artifact_router_scale_matches_the_helper` asserts the two agree
        // bit for bit rather than trusting this literal -- which is how the
        // first draft's wrong digits were caught.
        router_input_scale: 0.018_844_46,
    }),
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
    /// The routed-expert block, when the variant declares one.
    pub moe: Option<MoeGeometry>,
}

impl ArtifactGeometry {
    /// Whether layer `l` uses full causal attention.
    pub const fn global_layer(&self, layer: u32) -> bool {
        layer < self.layers && (layer + 1).is_multiple_of(self.global_stride)
    }
}

/// One layer type's key/value geometry, as the graph and the state both need it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerGeometry {
    pub kv_heads: u64,
    pub head_dim: u64,
    /// `None` on a global layer, the window on a sliding one.
    pub window: Option<u64>,
}

impl TextConfig {
    /// The key/value geometry and visibility of one layer.
    ///
    /// The single place the layer-type branch is resolved, so the graph, the
    /// admitted pages and any test all read the same answer rather than each
    /// re-deriving `(layer + 1) % stride`.
    pub const fn layer_geometry(&self, layer: u32) -> LayerGeometry {
        if self.global_layer(layer) {
            LayerGeometry {
                kv_heads: self.global_kv_heads,
                head_dim: self.global_head_dim,
                window: None,
            }
        } else {
            LayerGeometry {
                kv_heads: self.local_kv_heads,
                head_dim: self.local_head_dim,
                window: Some(self.sliding_window),
            }
        }
    }
}

/// What a reduced graph gives up, enumerated rather than described.
///
/// Task 0017 removed two of the four. Per-layer key/value geometry and
/// window reclamation are no longer reductions: a reduced graph now composes
/// its sliding and global layers at their own widths and its store keeps only
/// what each layer can see. The two that remain are the two that need a
/// checkpoint importer (M3) and a vision tower (M11), and neither is a
/// synthetic-scale difference that a smaller fixture could close.
///
/// Returned by [`Gemma4Text::reduction`] so that a diagnostic surface can print
/// it. A reduced graph that could not say what it reduced would be exactly the
/// "synthetic output described as model support" that document 06 forbids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reduction {
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
            // The routed branch. Every layer of the routed variant has all of
            // these *and* the dense `ffn_*` tensors above: the census over the
            // designated artifact's index is 30 of each across 30 layers.
            if config.moe.is_some() {
                for name in [
                    "router_scale",
                    "router_proj",
                    "router_per_expert_scale",
                    "experts_gate_up",
                    "experts_down",
                    "ffn_norm_2",
                    "ffn_out_norm_1",
                    "ffn_out_norm_2",
                ] {
                    tensors.push(required(name, Some(layer))?);
                }
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
        // Checked once per layer type rather than once, because the two types
        // no longer share a head dimension and so no longer share a query or
        // key/value width either.
        for (what, kv, dim) in [
            ("local", c.local_kv_heads, c.local_head_dim),
            ("global", c.global_kv_heads, c.global_head_dim),
        ] {
            let _ = what;
            width("heads * head_dim", c.heads, dim)?;
            width("kv_heads * head_dim", kv, dim)?;
        }
        let mut g = GraphBuilder::new(moxie_graph::OracleId("moxie_oracles::host_reference"), rows);
        let index = TensorSpec::new(
            ValueRole::Index(IndexEncoding::U64),
            vec![Dim::symbol(rows)],
        );
        let tokens = g.input("tokens", index.clone());
        let positions = g.input("absolute positions", index);

        let mut bound = Vec::new();

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
            // This layer's own widths. A graph that used one pair for every
            // layer would be describing a checkpoint whose sliding and global
            // layers agree, and Gemma 4's do not.
            let geometry = c.layer_geometry(layer);
            let head_dim = geometry.head_dim;
            let kv_heads = geometry.kv_heads;
            let query_width = width("heads * head_dim", c.heads, head_dim)?;
            let kv_width = width("kv_heads * head_dim", kv_heads, head_dim)?;
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

            let q_norm = weight(&mut g, &mut bound, "q_norm", Some(layer), vec![head_dim])?;
            let k_norm = weight(&mut g, &mut bound, "k_norm", Some(layer), vec![head_dim])?;
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
                vec![head_dim],
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
                    group: kv_heads,
                    eps: c.rms_eps,
                },
                &[k, k_norm],
            )?;
            let vn = g.node(
                OpParams::RmsNorm {
                    hidden: kv_width,
                    group: kv_heads,
                    eps: c.rms_eps,
                },
                &[v, v_gain],
            )?;

            let (theta, rotary) = if global {
                (c.global_rope_theta, c.global_partial_rotary.of(head_dim)?)
            } else {
                (c.sliding_rope_theta, head_dim)
            };
            let qr = g.node(
                OpParams::Rope {
                    heads: c.heads,
                    head_dim,
                    rotary_dim: rotary,
                    // The full head dimension, not the rotated width: a global
                    // layer rotates a quarter of the head and still divides by
                    // the whole of it.
                    frequency_dim: head_dim,
                    base: theta,
                    layout: RopeLayout::HalfSplit,
                },
                &[qn, positions],
            )?;
            let kr = g.node(
                OpParams::Rope {
                    heads: kv_heads,
                    head_dim,
                    rotary_dim: rotary,
                    frequency_dim: head_dim,
                    base: theta,
                    layout: RopeLayout::HalfSplit,
                },
                &[kn, positions],
            )?;

            let attention = g.node(
                OpParams::Attention {
                    heads: c.heads,
                    kv_heads,
                    head_dim,
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
            // The dense branch's output. On the routed variant this is the
            // shared expert's contribution, normalized on its own before it
            // meets the routed one; on the dense variant there is no second
            // branch and `feedforward` is this value directly.
            let feedforward = match c.moe {
                None => down,
                Some(moe) => route_layer(&mut g, &mut bound, c, layer, moe, stream, down)?,
            };
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
                &[feedforward, ffn_out_norm],
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

/// Declare a BF16 weight, label it and record the role it fills.
///
/// The model names roles and the composition root supplies bytes; neither knows
/// the other's file names. Shapes come from `graph.spec(value)`, so there is one
/// description of a tensor's extent and it is the graph's.
fn weight(
    g: &mut GraphBuilder,
    bound: &mut Vec<BoundRole>,
    name: &str,
    layer: Option<u32>,
    shape: Vec<u64>,
) -> Result<ValueId> {
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
}

/// One layer's routed-expert branch, combined with the dense shared expert.
///
/// `stream` is the **post-attention residual**, before `pre_feedforward_layernorm`;
/// `dense` is the dense MLP's output, before any normalization. Returns the
/// value `post_feedforward_layernorm` consumes.
///
/// Transcribed from `Gemma4TextDecoderLayer.forward` in the pinned
/// `transformers` source, where the order is:
///
/// ```text
/// h1 = post_feedforward_layernorm_1(mlp(pre_feedforward_layernorm(r)))
/// h2 = post_feedforward_layernorm_2(experts(pre_feedforward_layernorm_2(r), router(r)))
/// out = h1 + h2
/// ```
///
/// Two orderings in that are worth stating because a plausible graph gets them
/// wrong. **The router reads `r`, not the normalized stream** -- it is the only
/// consumer of the un-normalized residual in the block, and feeding it the
/// normalized one routes every row on a different vector while still producing
/// a well-shaped answer. And the **experts read a different normalization than
/// the dense MLP does**: `pre_feedforward_layernorm_2`, its own tensor, not the
/// one the shared expert used.
fn route_layer(
    g: &mut GraphBuilder,
    bound: &mut Vec<BoundRole>,
    c: &TextConfig,
    layer: u32,
    moe: MoeGeometry,
    stream: ValueId,
    dense: ValueId,
) -> Result<ValueId> {
    let norm = |g: &mut GraphBuilder, x: ValueId, gain: ValueId| -> Result<ValueId> {
        g.node(
            OpParams::RmsNorm {
                hidden: c.hidden,
                group: 1,
                eps: c.rms_eps,
            },
            &[x, gain],
        )
    };

    let out_norm_1 = weight(g, bound, "ffn_out_norm_1", Some(layer), vec![c.hidden])?;
    let shared = norm(g, dense, out_norm_1)?;

    // The router's three tensors. `router_scale` is the gain of the router's
    // own scale-free normalization, which is why it is bound as a weight here
    // rather than folded into a constant: the artifact stores a trained vector.
    let router_scale = weight(g, bound, "router_scale", Some(layer), vec![c.hidden])?;
    let router_proj = weight(
        g,
        bound,
        "router_proj",
        Some(layer),
        vec![moe.experts, c.hidden],
    )?;
    let per_expert = weight(
        g,
        bound,
        "router_per_expert_scale",
        Some(layer),
        vec![moe.experts],
    )?;
    let route = g.node(
        OpParams::Route {
            hidden: c.hidden,
            experts: moe.experts,
            top_k: moe.top_k,
            eps: c.rms_eps,
            input_scale: moe.router_input_scale,
            per_expert_scale: true,
        },
        &[stream, router_scale, router_proj, per_expert],
    )?;

    let ffn_norm_2 = weight(g, bound, "ffn_norm_2", Some(layer), vec![c.hidden])?;
    let routed_input = norm(g, stream, ffn_norm_2)?;
    // Fused across experts, exactly as the artifact stores them: one
    // `experts.gate_up_proj` and one `experts.down_proj` per layer holding all
    // of them, not one tensor per expert.
    let gate_up = weight(
        g,
        bound,
        "experts_gate_up",
        Some(layer),
        vec![moe.experts, 2 * moe.moe_intermediate, c.hidden],
    )?;
    let expert_down = weight(
        g,
        bound,
        "experts_down",
        Some(layer),
        vec![moe.experts, c.hidden, moe.moe_intermediate],
    )?;
    let slots = g.node(
        OpParams::ExpertMlp {
            hidden: c.hidden,
            intermediate: moe.moe_intermediate,
            experts: moe.experts,
            top_k: moe.top_k,
            // `hidden_activation` is `gelu_pytorch_tanh`, the same gate
            // transform the dense MLP above uses.
            activation: ExpertActivation::GeGlu,
        },
        &[routed_input, route, gate_up, expert_down],
    )?;
    let combined = g.node(
        OpParams::Combine {
            hidden: c.hidden,
            top_k: moe.top_k,
            // Ascending expert id: the pinned reference accumulates over
            // `expert_hit`, which is expert-major, not over the row's selection
            // order.
            order: CombineOrder::AscendingExpertId,
        },
        &[route, slots],
    )?;
    let out_norm_2 = weight(g, bound, "ffn_out_norm_2", Some(layer), vec![c.hidden])?;
    let routed = norm(g, combined, out_norm_2)?;

    // Unscaled, and unnormalized: the two branches are simply added. The layer
    // scalar belongs to the residual further down, and the dense branch takes
    // no routing coefficient -- it is a shared expert outside routing.
    g.node(OpParams::Residual { scale: 1.0 }, &[shared, routed])
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
        let mut ops = vec![
            Op::Embedding,
            Op::RmsNorm,
            Op::Linear,
            Op::Rope,
            Op::Attention,
            Op::Residual,
            Op::GeGlu,
            Op::VocabProjection,
        ];
        // Declared only when the variant actually has them, so that a dense
        // configuration cannot be admitted against a registry that happens to
        // carry routing references it will never use.
        if self.config.moe.is_some() {
            ops.extend([Op::Route, Op::ExpertMlp, Op::Combine]);
        }
        GraphRequirements { ops }
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
            local_kv_heads: 2,
            local_head_dim: 16,
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
            moe: None,
        }
    }

    /// The same reduced geometry with the designated artifact's routed block.
    fn routed_config() -> TextConfig {
        TextConfig {
            moe: Some(MoeGeometry {
                experts: 5,
                top_k: 2,
                moe_intermediate: 6,
                router_input_scale: router_input_scale(24),
            }),
            ..reduced_config()
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
                "local_kv_heads",
                |c: &mut TextConfig| c.local_kv_heads = 1 << 60,
                "kv_heads * head_dim",
            ),
            (
                "global_kv_heads",
                |c: &mut TextConfig| c.global_kv_heads = 1 << 60,
                "kv_heads * head_dim",
            ),
            (
                "local_head_dim",
                |c: &mut TextConfig| c.local_head_dim = 1 << 60,
                "heads * head_dim",
            ),
            (
                "global_head_dim",
                |c: &mut TextConfig| c.global_head_dim = 1 << 60,
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

    /// A registry naming a reference for every operation this crate composes.
    ///
    /// Built here rather than by depending on `moxie-oracles`: a model crate's
    /// allowed dependencies are graph, model-api and types, and reaching for
    /// the oracle crate even in a test would be the edge `arch-check` exists to
    /// refuse. The names are the ones `moxie-oracles` registers, so a graph
    /// that finishes here finishes there.
    fn oracles_registry() -> OracleRegistry {
        let mut o = OracleRegistry::new();
        for op in [
            Op::Embedding,
            Op::Linear,
            Op::RmsNorm,
            Op::Rope,
            Op::Attention,
            Op::Residual,
            Op::GeGlu,
            Op::VocabProjection,
            Op::Route,
            Op::ExpertMlp,
            Op::Combine,
        ] {
            o.register(
                op,
                moxie_graph::OracleId("moxie_oracles::host_reference"),
                moxie_graph::OracleEvidence {
                    implementation: "test",
                    test_module: "test",
                },
            )
            .unwrap();
        }
        o
    }

    #[test]
    fn the_a4b_artifact_geometry_matches_its_config_json() {
        // Read from `/fast/models/google/gemma-4-26B-A4B-it/config.json`,
        // revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, on 2026-09-12.
        // Transcribed here so that a later edit to the constant has to disagree
        // with the recorded inspection rather than with nothing.
        let a = ARTIFACT_A4B;
        assert_eq!((a.hidden, a.layers, a.heads), (2816, 30, 16));
        assert_eq!((a.local_kv_heads, a.local_head_dim), (8, 256));
        assert_eq!((a.global_kv_heads, a.global_head_dim), (2, 512));
        assert_eq!(a.intermediate, 2112);
        assert_eq!(a.vocab, 262_144);
        assert_eq!((a.global_stride, a.sliding_window), (6, 1024));
        assert_eq!(a.final_logit_softcap, 30.0);
        let moe = a.moe.expect("the A4B variant declares enable_moe_block");
        assert_eq!((moe.experts, moe.top_k), (128, 8));
        assert_eq!(moe.moe_intermediate, 704);

        // The same family as the 31B, which is why task 0017's per-layer
        // geometry and task 0016's operation parameters transfer rather than
        // being rebuilt: same predicate, same asymmetry, same softcap, same
        // window. The five full-attention layers are 5, 11, 17, 23, 29.
        let global: Vec<u32> = (0..a.layers).filter(|l| a.global_layer(*l)).collect();
        assert_eq!(global, vec![5, 11, 17, 23, 29]);
        assert_eq!(a.global_stride, ARTIFACT.global_stride);
        assert_eq!(a.final_logit_softcap, ARTIFACT.final_logit_softcap);
        assert_eq!(a.sliding_window, ARTIFACT.sliding_window);
        assert!(ARTIFACT.moe.is_none(), "the 31B is dense");
    }

    #[test]
    fn the_artifact_router_scale_matches_the_helper() {
        // A `const` cannot call a function, so the literal is checked against
        // the helper bit for bit. The first draft of that literal was wrong in
        // its fifth significant digit and this is what caught it.
        let moe = ARTIFACT_A4B.moe.unwrap();
        assert_eq!(
            moe.router_input_scale.to_bits(),
            router_input_scale(ARTIFACT_A4B.hidden).to_bits(),
            "{} vs {}",
            moe.router_input_scale,
            router_input_scale(ARTIFACT_A4B.hidden)
        );
        // It is `hidden^(-1/2)` and nothing else -- not an epsilon, not the
        // embedding's `sqrt(hidden)`, not an attention scale.
        assert!((router_input_scale(2816) as f64 * (2816f64).sqrt() - 1.0).abs() < 1e-6);
        assert_ne!(router_input_scale(24), embedding_scale(24));
    }

    #[test]
    fn a_routed_layer_declares_the_routed_roles_beside_the_dense_ones() {
        let model = Gemma4Text::reduced(routed_config(), "synthetic-routed").unwrap();
        let names: Vec<(&str, Option<u32>)> = model
            .tensors()
            .iter()
            .map(|t| (t.role.name.as_str(), t.role.layer))
            .collect();
        for layer in 0..routed_config().layers {
            for role in [
                "router_scale",
                "router_proj",
                "router_per_expert_scale",
                "experts_gate_up",
                "experts_down",
                "ffn_norm_2",
                "ffn_out_norm_1",
                "ffn_out_norm_2",
                // Beside, never instead of: every layer of the designated
                // artifact carries a dense `mlp` as well as its experts, and a
                // graph that dropped it would be missing a shared expert.
                "ffn_gate",
                "ffn_up",
                "ffn_down",
                "ffn_norm",
                "ffn_out_norm",
            ] {
                assert!(
                    names.contains(&(role, Some(layer))),
                    "layer {layer} is missing {role}"
                );
            }
        }
    }

    #[test]
    fn a_dense_configuration_declares_no_routed_roles() {
        let model = Gemma4Text::reduced(reduced_config(), "synthetic").unwrap();
        for t in model.tensors() {
            assert!(
                !t.role.name.starts_with("router") && !t.role.name.starts_with("experts"),
                "a dense variant declared {}",
                t.role.name
            );
        }
        assert!(!model.graph_requirements().ops.contains(&Op::Route));
    }

    #[test]
    fn the_routed_graph_uses_the_shared_routing_operations_once_per_layer() {
        let config = routed_config();
        let model = Gemma4Text::reduced(config.clone(), "synthetic-routed").unwrap();
        let composed = model.compose(&oracles_registry(), SymbolId(0)).unwrap();
        for op in [Op::Route, Op::ExpertMlp, Op::Combine] {
            let count = composed
                .graph
                .nodes()
                .iter()
                .filter(|n| n.params.op() == op)
                .count();
            assert_eq!(count, config.layers as usize, "{} nodes", op.name());
            assert!(model.graph_requirements().ops.contains(&op));
        }

        // The fused expert tensors carry the artifact's shape: all experts in
        // one tensor, gate and up stacked along the output axis.
        let moe = config.moe.unwrap();
        let shape = |name: &str| -> Vec<Dim> {
            let bound = composed
                .weights
                .iter()
                .find(|b| b.role.name == name && b.role.layer == Some(0))
                .unwrap();
            composed.graph.spec(bound.value).unwrap().shape.clone()
        };
        assert_eq!(
            shape("experts_gate_up"),
            vec![
                Dim::constant(moe.experts),
                Dim::constant(2 * moe.moe_intermediate),
                Dim::constant(config.hidden),
            ]
        );
        assert_eq!(
            shape("experts_down"),
            vec![
                Dim::constant(moe.experts),
                Dim::constant(config.hidden),
                Dim::constant(moe.moe_intermediate),
            ]
        );
    }

    #[test]
    fn the_router_reads_the_residual_and_the_experts_read_their_own_norm() {
        // The ordering a plausible graph gets wrong. In the pinned reference
        // the router's argument is `residual`, taken *before*
        // `pre_feedforward_layernorm`; the experts' argument is
        // `pre_feedforward_layernorm_2(residual)`, its own tensor. A graph that
        // fed the router the normalized stream is well shaped and routes every
        // row on a different vector.
        let config = routed_config();
        let model = Gemma4Text::reduced(config, "synthetic-routed").unwrap();
        let composed = model.compose(&oracles_registry(), SymbolId(0)).unwrap();
        let nodes = composed.graph.nodes();
        let producer = |v: ValueId| nodes.iter().find(|n| n.output == v);

        let ffn_norm_2: Vec<ValueId> = composed
            .weights
            .iter()
            .filter(|b| b.role.name == "ffn_norm_2")
            .map(|b| b.value)
            .collect();

        for node in nodes.iter().filter(|n| n.params.op() == Op::Route) {
            let source = producer(node.inputs[0]).expect("the router reads a computed value");
            assert_eq!(
                source.params.op(),
                Op::Residual,
                "the router read a {} instead of the residual stream",
                source.params.op().name()
            );
        }
        for node in nodes.iter().filter(|n| n.params.op() == Op::ExpertMlp) {
            let source = producer(node.inputs[0]).expect("the experts read a computed value");
            assert_eq!(source.params.op(), Op::RmsNorm);
            assert!(
                ffn_norm_2.contains(&source.inputs[1]),
                "the experts were normalized with the dense branch's gain"
            );
            // And that norm reads the residual, not the dense branch's input.
            let normed = producer(source.inputs[0]).unwrap();
            assert_eq!(normed.params.op(), Op::Residual);
        }
    }

    #[test]
    fn the_routed_and_dense_branches_use_different_intermediate_widths() {
        // 704 against 2112 in the artifact. Equal widths here would let a graph
        // that confused the two branches pass every shape check.
        let config = routed_config();
        let moe = config.moe.unwrap();
        assert_ne!(moe.moe_intermediate, config.intermediate);
        let a = ARTIFACT_A4B;
        assert_ne!(a.moe.unwrap().moe_intermediate, a.intermediate);
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
