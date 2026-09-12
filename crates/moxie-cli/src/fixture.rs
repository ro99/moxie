//! Synthetic mathematical fixtures for the diagnostic composition root. No
//! checkpoint, tokenizer, model family or execution ownership lives here.
use moxie_engine::{HostTensor, Program, Value};
use moxie_graph::{
    Bindings, CombineOrder, ExpertActivation, Graph, GraphBuilder, IndexEncoding, OpParams,
    OracleRegistry, RopeLayout, TensorSpec, ValueId, ValueRole, Visibility, reciprocal_sqrt_scale,
};
use moxie_types::{Dim, Precision, Result, SymbolId, WeightPrecision};

#[derive(Debug)]
pub struct Fixture {
    pub graph: Graph,
    pub weights: Bindings<Value>,
    pub tokens: ValueId,
    pub positions: ValueId,
}
impl Fixture {
    pub fn program(&self) -> Program<'_> {
        Program {
            graph: &self.graph,
            weights: &self.weights,
            tokens: self.tokens,
            positions: self.positions,
        }
    }
}

pub fn build(heads: u64, dim: u64, vocab: u64, layers: u32) -> Result<Fixture> {
    build_with_rows(heads, dim, vocab, layers, Dim::symbol(SymbolId(0)))
}

/// Review fixture for admission: a fixed-row graph may be legal for its first
/// prefill chunk but cannot serve a tail or one-row decode.
pub fn build_fixed_rows(
    heads: u64,
    dim: u64,
    vocab: u64,
    layers: u32,
    rows: u64,
) -> Result<Fixture> {
    build_with_rows(heads, dim, vocab, layers, Dim::constant(rows))
}

fn build_with_rows(heads: u64, dim: u64, vocab: u64, layers: u32, rows: Dim) -> Result<Fixture> {
    let width = heads * dim;
    let mut graph = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let input = TensorSpec::new(ValueRole::Index(IndexEncoding::U64), vec![rows]);
    let tokens = graph.input("tokens", input.clone());
    let positions = graph.input("absolute positions", input);
    let mut weights = Bindings::new();
    let mut parameter =
        |g: &mut GraphBuilder, name: &str, rows: u64, cols: u64, offset: u64| -> Result<ValueId> {
            let id = g.weight(
                name,
                TensorSpec::new(
                    ValueRole::Weight(WeightPrecision::new(Precision::Bf16)?),
                    vec![Dim::constant(rows), Dim::constant(cols)],
                ),
            )?;
            let data = (0..rows * cols)
                .map(|i| (((i * 13 + offset) % 31) as f32 - 15.0) / 32.0)
                .collect();
            weights.set(
                id,
                Value::Float(HostTensor::bf16(data, vec![rows as usize, cols as usize])?),
            );
            Ok(id)
        };
    let embedding = parameter(&mut graph, "synthetic embeddings", vocab, width, 3)?;
    let projection = parameter(
        &mut graph,
        "synthetic vocabulary projection",
        vocab,
        width,
        11,
    )?;
    let mut value = graph.node(
        OpParams::Embedding {
            vocab,
            hidden: width,
            scale: 1.0,
        },
        &[tokens, embedding],
    )?;
    for layer in 0..layers {
        let rotated = graph.node(
            OpParams::Rope {
                heads,
                head_dim: dim,
                rotary_dim: dim,
                base: 10_000.0,
                frequency_dim: dim,
                layout: RopeLayout::Interleaved,
            },
            &[value, positions],
        )?;
        let attention = graph.node(
            OpParams::Attention {
                heads,
                head_dim: dim,
                visibility: Visibility::Causal,
                layer,
                kv_heads: heads,
                scale: reciprocal_sqrt_scale(dim),
            },
            &[rotated, rotated, value, positions],
        )?;
        value = graph.node(OpParams::Residual { scale: 1.0 }, &[value, attention])?;
    }
    let logits = graph.node(
        OpParams::VocabProjection {
            hidden: width,
            vocab,
            softcap: None,
        },
        &[value, projection],
    )?;
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles)?;
    Ok(Fixture {
        graph: graph.finish(logits, &oracles)?,
        weights,
        tokens,
        positions,
    })
}

/// A synthetic routed-expert graph, deliberately unlike the Gemma-like one.
///
/// Document 02's extension rule: "test a second existing consumer if one
/// exists, otherwise an independent synthetic consumer with different shapes".
/// Roadmap M2 item 4 asks for the same thing in its own words -- "a second
/// synthetic MoE consumer with different expert count, activation, top-k,
/// shapes and route distribution". Every routing parameter here carries the
/// opposite value to the Gemma-like graph's:
///
/// | | Gemma-like (`gemma-c-moe`) | this graph |
/// |---|---|---|
/// | experts / top-k | 5 / 2 | 3 / 3, so `top_k == experts` |
/// | activation | GeGLU | SwiGLU |
/// | per-expert coefficient scale | present | absent |
/// | combination order | ascending expert id | selection order |
/// | shared expert | a dense MLP beside the routed branch | none |
/// | router input scale | `hidden^(-1/2)` | 1.0 |
///
/// `top_k == experts` is the case worth naming: the renormalisation becomes a
/// no-op over the whole distribution, so a selection that silently dropped an
/// expert, or a renormalisation that divided by the wrong mass, shows up as a
/// changed result rather than as nothing.
pub fn build_routed() -> Result<Fixture> {
    const HEADS: u64 = 2;
    const DIM: u64 = 4;
    const WIDTH: u64 = HEADS * DIM;
    const VOCAB: u64 = 9;
    const EXPERTS: u64 = 3;
    const TOP_K: u64 = 3;
    const INTERMEDIATE: u64 = 5;

    let rows = Dim::symbol(SymbolId(0));
    let mut graph = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let input = TensorSpec::new(ValueRole::Index(IndexEncoding::U64), vec![rows]);
    let tokens = graph.input("tokens", input.clone());
    let positions = graph.input("absolute positions", input);
    let mut weights = Bindings::new();
    let mut parameter =
        |g: &mut GraphBuilder, name: &str, shape: Vec<u64>, offset: u64| -> Result<ValueId> {
            let id = g.weight(
                name,
                TensorSpec::new(
                    ValueRole::Weight(WeightPrecision::new(Precision::Bf16)?),
                    shape.iter().map(|d| Dim::constant(*d)).collect(),
                ),
            )?;
            let count: u64 = shape.iter().product();
            let data = (0..count)
                .map(|i| (((i * 17 + offset) % 23) as f32 - 11.0) / 32.0)
                .collect();
            weights.set(
                id,
                Value::Float(HostTensor::bf16(
                    data,
                    shape.iter().map(|d| *d as usize).collect(),
                )?),
            );
            Ok(id)
        };

    let embedding = parameter(&mut graph, "synthetic embeddings", vec![VOCAB, WIDTH], 3)?;
    let projection = parameter(
        &mut graph,
        "synthetic vocabulary projection",
        vec![VOCAB, WIDTH],
        11,
    )?;
    let router_gain = parameter(&mut graph, "router gain", vec![WIDTH], 5)?;
    let router_proj = parameter(&mut graph, "router projection", vec![EXPERTS, WIDTH], 7)?;
    let gate_up = parameter(
        &mut graph,
        "fused expert gate/up",
        vec![EXPERTS, 2 * INTERMEDIATE, WIDTH],
        2,
    )?;
    let down = parameter(
        &mut graph,
        "fused expert down",
        vec![EXPERTS, WIDTH, INTERMEDIATE],
        13,
    )?;

    let mut value = graph.node(
        OpParams::Embedding {
            vocab: VOCAB,
            hidden: WIDTH,
            scale: 1.0,
        },
        &[tokens, embedding],
    )?;
    let rotated = graph.node(
        OpParams::Rope {
            heads: HEADS,
            head_dim: DIM,
            rotary_dim: DIM,
            base: 10_000.0,
            frequency_dim: DIM,
            layout: RopeLayout::Interleaved,
        },
        &[value, positions],
    )?;
    let attention = graph.node(
        OpParams::Attention {
            heads: HEADS,
            head_dim: DIM,
            visibility: Visibility::Causal,
            layer: 0,
            kv_heads: HEADS,
            scale: reciprocal_sqrt_scale(DIM),
        },
        &[rotated, rotated, value, positions],
    )?;
    value = graph.node(OpParams::Residual { scale: 1.0 }, &[value, attention])?;

    // No shared expert and no per-expert scale: the routed branch is the whole
    // feed-forward here, which is what makes it a different consumer rather
    // than the same block with different numbers.
    let route = graph.node(
        OpParams::Route {
            hidden: WIDTH,
            experts: EXPERTS,
            top_k: TOP_K,
            eps: 1e-5,
            input_scale: 1.0,
            per_expert_scale: false,
        },
        &[value, router_gain, router_proj],
    )?;
    let slots = graph.node(
        OpParams::ExpertMlp {
            hidden: WIDTH,
            intermediate: INTERMEDIATE,
            experts: EXPERTS,
            top_k: TOP_K,
            activation: ExpertActivation::SwiGlu,
        },
        &[value, route, gate_up, down],
    )?;
    let combined = graph.node(
        OpParams::Combine {
            hidden: WIDTH,
            top_k: TOP_K,
            order: CombineOrder::SelectionOrder,
        },
        &[route, slots],
    )?;
    value = graph.node(OpParams::Residual { scale: 1.0 }, &[value, combined])?;

    let logits = graph.node(
        OpParams::VocabProjection {
            hidden: WIDTH,
            vocab: VOCAB,
            softcap: None,
        },
        &[value, projection],
    )?;
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles)?;
    Ok(Fixture {
        graph: graph.finish(logits, &oracles)?,
        weights,
        tokens,
        positions,
    })
}
