//! Synthetic mathematical fixtures for the diagnostic composition root. No
//! checkpoint, tokenizer, model family or execution ownership lives here.
use moxie_engine::{HostTensor, Program, Value};
use moxie_graph::{
    Bindings, Graph, GraphBuilder, IndexEncoding, OpParams, OracleRegistry, TensorSpec, ValueId,
    ValueRole, Visibility,
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
            },
            &[value, positions],
        )?;
        let attention = graph.node(
            OpParams::Attention {
                heads,
                head_dim: dim,
                visibility: Visibility::Causal,
                layer,
            },
            &[rotated, rotated, value, positions],
        )?;
        value = graph.node(OpParams::Residual, &[value, attention])?;
    }
    let logits = graph.node(
        OpParams::VocabProjection {
            hidden: width,
            vocab,
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
