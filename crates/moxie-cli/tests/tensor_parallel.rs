//! Host tensor-parallel execution of the attention sublayer (task 0056).
//!
//! **Temporary harness.** The multi-rank runner and the host gather below
//! stand in for executor collectives. They expire when the device TP2 slice
//! lands, and the stage subgraph helper must not become a production API.
//!
//! Each [`Stage`] runs as its own small graph. Replicated stages go through
//! `run_stateless`. A head-local stage goes through `Interpreter::run` with
//! the rank's own one-layer state for that layer. So one rank step is one
//! transaction **per layer**, not one per step: a failure part-way through a
//! step would leave earlier layers appended. The device slice owns making a
//! whole rank step atomic.

use std::collections::BTreeMap;
use std::ops::Range;

use moxie_cli::{fixture::Fixture, gemma};
use moxie_engine::{Cancel, HostTensor, Value};
use moxie_graph::{
    Bindings, Graph, GraphBuilder, IndexEncoding, OpParams, OracleRegistry, TensorSpec, ValueId,
    ValueRole, Visibility,
};
use moxie_interp::{Interpreter, KvCache};
use moxie_plan::{
    RankPart, Stage, TensorParallelLowering, TensorParallelRefused, lower_tensor_parallel,
};
use moxie_state::{ROOT, SequenceState, StateKind};
use moxie_types::{Dim, Precision, SymbolId, WeightPrecision};

/// Shape A's geometry with 8 query heads over 4 sliding and 1 global
/// key/value heads: at 2 and 4 ranks the sliding layers split their key/value
/// heads, at 8 each sliding key/value head is replicated over two ranks, and
/// the global layer always replicates its one.
fn fixture() -> Fixture {
    let mut config = gemma::Shape::A.config();
    config.heads = 8;
    config.local_kv_heads = 4;
    config.global_kv_heads = 1;
    gemma::build_with_config(config).unwrap()
}

/// One stage as a graph of its own, with the original ids of the values it
/// reads from outside and the original id of each value it produces.
struct StageGraph {
    graph: Graph,
    bindings: Bindings<Value>,
    reads: Vec<(ValueId, ValueId)>,
    produces: Vec<(ValueId, ValueId)>,
}

/// Rebuild `nodes` of `graph` as a standalone graph. `part` supplies a rank's
/// parameters and weight rows; head-local attention is renumbered to layer 0,
/// because the stage graph writes exactly one cache layer. `output`, when
/// given, is the value the stage graph returns; otherwise its last node's.
fn stage_graph(
    graph: &Graph,
    weights: &Bindings<Value>,
    nodes: Range<usize>,
    part: Option<&RankPart>,
    output: Option<ValueId>,
) -> StageGraph {
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let mut builder = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, graph.rows_symbol());
    let mut map: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    let mut bindings = Bindings::new();
    let mut reads = Vec::new();
    let mut produces = Vec::new();
    let mut last = None;
    for node in &graph.nodes()[nodes] {
        let mut inputs = Vec::new();
        for &input in &node.inputs {
            let id = *map.entry(input).or_insert_with(|| {
                let name = graph.name(input).unwrap_or("value");
                let mut spec = graph.spec(input).unwrap().clone();
                if graph.weights().contains(&input) {
                    let mut value = weights.get(input).unwrap().clone();
                    if let Some(rows) = part.and_then(|p| p.rows.get(&input)) {
                        let whole = value.as_float().unwrap();
                        let cols = whole.cols();
                        let (start, end) = (rows.start as usize, rows.end as usize);
                        spec.shape[0] = Dim::constant(rows.end - rows.start);
                        let data = whole.data()[start * cols..end * cols].to_vec();
                        value =
                            Value::Float(HostTensor::bf16(data, vec![end - start, cols]).unwrap());
                    }
                    let id = builder.weight(name, spec).unwrap();
                    bindings.set(id, value);
                    id
                } else {
                    let id = builder.input(name, spec);
                    reads.push((input, id));
                    id
                }
            });
            inputs.push(id);
        }
        let params = match part.and_then(|p| p.params.get(&node.id)).cloned() {
            Some(OpParams::Attention {
                heads,
                kv_heads,
                head_dim,
                scale,
                visibility,
                ..
            }) => OpParams::Attention {
                heads,
                kv_heads,
                head_dim,
                scale,
                visibility,
                layer: 0,
            },
            Some(params) => params,
            None => node.params.clone(),
        };
        let out = builder.node(params, &inputs).unwrap();
        map.insert(node.output, out);
        produces.push((node.output, out));
        last = Some(out);
    }
    StageGraph {
        graph: builder
            .finish(output.map_or(last.unwrap(), |o| map[&o]), &oracles)
            .unwrap(),
        bindings,
        reads,
        produces,
    }
}

fn bind(stage: &StageGraph, table: &BTreeMap<ValueId, Value>) -> Bindings<Value> {
    let mut bindings = stage.bindings.clone();
    for (original, id) in &stage.reads {
        bindings.set(*id, table[original].clone());
    }
    bindings
}

fn bits(tensor: &HostTensor) -> Vec<u32> {
    tensor.data().iter().map(|x| x.to_bits()).collect()
}

/// Concatenate row-major tensors along columns, in the given order.
fn concatenate_columns(parts: &[HostTensor]) -> HostTensor {
    let rows = parts[0].rows();
    let mut data = Vec::new();
    for row in 0..rows {
        for part in parts {
            data.extend_from_slice(part.row(row).unwrap());
        }
    }
    let cols = parts.iter().map(HostTensor::cols).sum();
    HostTensor::bf16(data, vec![rows, cols]).unwrap()
}

/// The steps both runs take: a multi-row prefill, then two one-row decodes.
const STEPS: [Range<u64>; 3] = [0..5, 5..6, 6..7];

fn tokens(rows: &Range<u64>, vocab: u64) -> Vec<u64> {
    rows.clone().map(|i| (i * 3 + 1) % vocab).collect()
}

fn unsplit_logits(f: &Fixture, vocab: u64) -> Vec<Vec<u32>> {
    let layers = f.graph.attention_layers().len();
    let mut state = SequenceState::new([StateKind::KvPages]);
    let mut cache = KvCache::for_branch(layers, &state, ROOT).unwrap();
    STEPS
        .iter()
        .map(|rows| {
            state.append_prompt(ROOT, rows.end - rows.start).unwrap();
            let mut bindings = f.weights.clone();
            bindings.set(f.tokens, Value::Index(tokens(rows, vocab)));
            bindings.set(f.positions, Value::Index(rows.clone().collect()));
            let out = Interpreter::new()
                .run(
                    &f.graph,
                    &bindings,
                    &mut state,
                    ROOT,
                    &mut cache,
                    &Cancel::never(),
                )
                .unwrap();
            bits(&out.logits)
        })
        .collect()
}

fn split_logits(f: &Fixture, lowering: &TensorParallelLowering, vocab: u64) -> Vec<Vec<u32>> {
    let ranks = lowering.ranks.len();
    // Per stage, its gather value when head-local, and one graph per rank.
    let stages: Vec<(Option<ValueId>, Vec<StageGraph>)> = lowering
        .stages
        .iter()
        .map(|stage| match stage {
            Stage::Replicated(nodes) => {
                let g = || stage_graph(&f.graph, &f.weights, nodes.clone(), None, None);
                (None, (0..ranks).map(|_| g()).collect())
            }
            Stage::HeadLocal { nodes, gather } => (
                Some(*gather),
                lowering
                    .ranks
                    .iter()
                    .map(|part| {
                        stage_graph(
                            &f.graph,
                            &f.weights,
                            nodes.clone(),
                            Some(part),
                            Some(*gather),
                        )
                    })
                    .collect(),
            ),
        })
        .collect();
    // One state and one one-layer cache per (head-local stage, rank).
    let mut kv: Vec<Vec<(SequenceState, KvCache)>> = stages
        .iter()
        .filter(|(gather, _)| gather.is_some())
        .map(|_| {
            (0..ranks)
                .map(|_| {
                    let state = SequenceState::new([StateKind::KvPages]);
                    let cache = KvCache::for_branch(1, &state, ROOT).unwrap();
                    (state, cache)
                })
                .collect()
        })
        .collect();
    let interpreter = Interpreter::new();
    STEPS
        .iter()
        .map(|rows| {
            let mut tables: Vec<BTreeMap<ValueId, Value>> = (0..ranks)
                .map(|_| {
                    BTreeMap::from([
                        (f.tokens, Value::Index(tokens(rows, vocab))),
                        (f.positions, Value::Index(rows.clone().collect())),
                    ])
                })
                .collect();
            let mut layer = 0;
            for (gather, graphs) in &stages {
                if let Some(gather) = *gather {
                    let mut outputs = Vec::new();
                    for (rank, stage) in graphs.iter().enumerate() {
                        let (state, cache) = &mut kv[layer][rank];
                        state.append_prompt(ROOT, rows.end - rows.start).unwrap();
                        let out = interpreter
                            .run(
                                &stage.graph,
                                &bind(stage, &tables[rank]),
                                state,
                                ROOT,
                                cache,
                                &Cancel::never(),
                            )
                            .unwrap();
                        outputs.push(out.logits);
                    }
                    let gathered = Value::Float(concatenate_columns(&outputs));
                    for table in &mut tables {
                        table.insert(gather, gathered.clone());
                    }
                    layer += 1;
                } else {
                    for (rank, stage) in graphs.iter().enumerate() {
                        let trace = interpreter
                            .run_stateless(&stage.graph, &bind(stage, &tables[rank]))
                            .unwrap();
                        for (original, id) in &stage.produces {
                            let value = trace.node_output(*id).unwrap().clone();
                            tables[rank].insert(*original, value);
                        }
                    }
                    for table in &tables[1..] {
                        for (original, _) in &graphs[0].produces {
                            let (a, b) = (&tables[0][original], &table[original]);
                            assert_eq!(
                                bits(a.as_float().unwrap()),
                                bits(b.as_float().unwrap()),
                                "replicated value {} differs across ranks",
                                original.0
                            );
                        }
                    }
                }
            }
            bits(tables[0][&f.graph.output()].as_float().unwrap())
        })
        .collect()
}

#[test]
fn head_split_attention_is_bit_identical_to_the_unsplit_graph() {
    let f = fixture();
    let vocab = gemma::Shape::A.config().vocab;
    let reference = unsplit_logits(&f, vocab);
    for ranks in [2, 4, 8] {
        let lowering = lower_tensor_parallel(&f.graph, ranks).unwrap();
        assert_eq!(
            split_logits(&f, &lowering, vocab),
            reference,
            "{ranks} ranks"
        );
    }
}

#[test]
fn unsupported_partitions_are_refused() {
    type Expect = fn(&TensorParallelRefused) -> bool;
    let cases: [(&str, Graph, u32, Expect); 4] = [
        (
            "4 query heads over 3 ranks",
            gemma::build(gemma::Shape::A).unwrap().graph,
            3,
            |r| matches!(r, TensorParallelRefused::Heads { .. }),
        ),
        (
            "3 global kv heads over 2 ranks",
            gemma::build(gemma::Shape::B).unwrap().graph,
            2,
            |r| matches!(r, TensorParallelRefused::KvHeads { .. }),
        ),
        (
            "a routed graph",
            gemma::build(gemma::Shape::C).unwrap().graph,
            2,
            |r| matches!(r, TensorParallelRefused::Op { .. }),
        ),
        ("a biased query projection", biased_query_graph(), 2, |r| {
            matches!(r, TensorParallelRefused::HeadChain { .. })
        }),
    ];
    for (what, graph, ranks, expected) in cases {
        let refused = lower_tensor_parallel(&graph, ranks).unwrap_err();
        assert!(expected(&refused), "{what}: {refused}");
    }
}

/// One attention layer whose query projection carries a bias.
fn biased_query_graph() -> Graph {
    let (heads, head_dim, hidden, vocab) = (2u64, 4u64, 8u64, 5u64);
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let mut g = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let index = TensorSpec::new(
        ValueRole::Index(IndexEncoding::U64),
        vec![Dim::symbol(SymbolId(0))],
    );
    let tokens = g.input("tokens", index.clone());
    let positions = g.input("positions", index);
    let weight = |g: &mut GraphBuilder, shape: Vec<u64>| {
        let spec = TensorSpec::new(
            ValueRole::Weight(WeightPrecision::new(Precision::Bf16).unwrap()),
            shape.into_iter().map(Dim::constant).collect(),
        );
        g.weight("w", spec).unwrap()
    };
    let table = weight(&mut g, vec![vocab, hidden]);
    let x = g
        .node(
            OpParams::Embedding {
                vocab,
                hidden,
                scale: 1.0,
            },
            &[tokens, table],
        )
        .unwrap();
    let width = heads * head_dim;
    let linear = |g: &mut GraphBuilder, bias: bool| {
        let w = weight(g, vec![width, hidden]);
        let mut inputs = vec![x, w];
        if bias {
            inputs.push(weight(g, vec![width]));
        }
        let params = OpParams::Linear {
            in_features: hidden,
            out_features: width,
            bias,
        };
        g.node(params, &inputs).unwrap()
    };
    let (q, k, v) = (
        linear(&mut g, true),
        linear(&mut g, false),
        linear(&mut g, false),
    );
    let attention = OpParams::Attention {
        heads,
        kv_heads: heads,
        head_dim,
        scale: 1.0,
        visibility: Visibility::Causal,
        layer: 0,
    };
    let a = g.node(attention, &[q, k, v, positions]).unwrap();
    g.finish(a, &oracles).unwrap()
}
