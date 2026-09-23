//! Host tensor-parallel execution of the dense M5 slice.
//!
//! **Temporary harness.** The multi-rank runner and host joins below stand in
//! for executor collectives. They expire when the device TP2 slice lands, and
//! the stage subgraph helper must not become a production API.

use std::collections::BTreeMap;
use std::ops::Range;

use moxie_cli::{fixture::Fixture, gemma};
use moxie_engine::{Cancel, HostTensor, Value};
use moxie_graph::{
    Bindings, Graph, GraphBuilder, IndexEncoding, LinearInputSlice, LinearReductionOrder, NodeId,
    OpParams, OracleRegistry, TensorSpec, ValueId, ValueRole, Visibility,
};
use moxie_interp::{Interpreter, KvCache};
use moxie_plan::{
    Join, RankPart, Stage, TensorParallelLowering, TensorParallelRefused, lower_tensor_parallel,
};
use moxie_state::{ROOT, SequenceState, StateKind};
use moxie_types::{ActivationPrecision, Dim, Precision, SymbolId, WeightPrecision};

/// Shape A's geometry with 8 query heads over 4 sliding and 1 global
/// key/value heads. The vocabulary is made divisible by both accepted dense
/// TP rank counts so the vocabulary gather is exercised as well.
fn fixture() -> Fixture {
    let mut config = gemma::Shape::A.config();
    config.heads = 8;
    config.local_kv_heads = 4;
    config.global_kv_heads = 1;
    config.vocab = 12;
    gemma::build_with_config(config).unwrap()
}

/// One stage as a graph of its own, with original value ids for its boundary.
struct StageGraph {
    graph: Graph,
    bindings: Bindings<Value>,
    reads: Vec<(ValueId, ValueId, Option<LinearInputSlice>)>,
    produces: Vec<(ValueId, ValueId)>,
    linear_orders: BTreeMap<NodeId, LinearReductionOrder>,
}

/// Rebuild a consecutive stage as a standalone graph. A local stage supplies
/// compact weight rows/input columns and a node-keyed reduction order.
fn stage_graph(
    graph: &Graph,
    weights: &Bindings<Value>,
    nodes: Range<usize>,
    part: Option<&RankPart>,
    output: Option<ValueId>,
) -> StageGraph {
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let built = moxie_plan::build_stage_graph(
        graph,
        part,
        nodes,
        output,
        moxie_oracles::HOST_REFERENCE,
        &oracles,
    )
    .unwrap();
    let mut bindings = Bindings::new();
    for weight in &built.weights {
        let mut value = weights.get(weight.original).unwrap().clone();
        if let Some(rows) = &weight.rows {
            let whole = value.as_float().unwrap();
            let cols = whole.cols();
            let (start, end) = (rows.start as usize, rows.end as usize);
            let data = whole.data()[start * cols..end * cols].to_vec();
            value = Value::Float(HostTensor::bf16(data, vec![end - start, cols]).unwrap());
        } else if let Some(slice) = weight.slice {
            let whole = value.as_float().unwrap();
            let rows = whole.rows();
            let cols = whole.cols();
            let start = slice.first as usize;
            let end = start + slice.width as usize;
            let mut data = Vec::with_capacity(rows * (end - start));
            for row in 0..rows {
                data.extend_from_slice(&whole.data()[row * cols + start..row * cols + end]);
            }
            value = Value::Float(HostTensor::bf16(data, vec![rows, end - start]).unwrap());
        }
        bindings.set(weight.local, value);
    }
    StageGraph {
        graph: built.graph,
        bindings,
        reads: built
            .reads
            .into_iter()
            .map(|read| (read.original, read.local, read.slice))
            .collect(),
        produces: built.produces,
        linear_orders: built.linear_orders,
    }
}

fn bind(stage: &StageGraph, table: &BTreeMap<ValueId, Value>) -> Bindings<Value> {
    let mut bindings = stage.bindings.clone();
    for (original, id, slice) in &stage.reads {
        let mut value = table[original].clone();
        if let Some(slice) = slice {
            let whole = value.as_float().unwrap();
            let cols = whole.cols();
            let start = slice.first as usize;
            let end = start + slice.width as usize;
            let mut data = Vec::with_capacity(whole.rows() * (end - start));
            for row in 0..whole.rows() {
                data.extend_from_slice(&whole.data()[row * cols + start..row * cols + end]);
            }
            value = Value::Float(HostTensor::bf16(data, vec![whole.rows(), end - start]).unwrap());
        }
        bindings.set(*id, value);
    }
    bindings
}

fn bits(tensor: &HostTensor) -> Vec<u32> {
    tensor.data().iter().map(|x| x.to_bits()).collect()
}

/// Concatenate row-major tensors along columns, preserving FP32 vocabulary
/// logits and BF16 activation boundaries.
fn concatenate_columns(parts: &[HostTensor]) -> HostTensor {
    let rows = parts[0].rows();
    let mut data = Vec::new();
    for row in 0..rows {
        for part in parts {
            data.extend_from_slice(part.row(row).unwrap());
        }
    }
    let cols = parts.iter().map(HostTensor::cols).sum();
    if parts[0].precision() == Precision::F32 {
        HostTensor::f32(data, vec![rows, cols]).unwrap()
    } else {
        HostTensor::bf16(data, vec![rows, cols]).unwrap()
    }
}

/// Combine rank-local FP32 partials in ascending rank order, then cross the
/// single BF16 boundary for the row-parallel Linear output.
fn reduce_rows(parts: &[HostTensor]) -> HostTensor {
    let rows = parts[0].rows();
    let cols = parts[0].cols();
    let mut data = vec![0.0; rows * cols];
    for part in parts {
        assert_eq!(part.precision(), Precision::F32);
        for (sum, value) in data.iter_mut().zip(part.data()) {
            *sum += value;
        }
    }
    HostTensor::round_to_bf16(data, vec![rows, cols]).unwrap()
}

const STEPS: [Range<u64>; 3] = [0..5, 5..6, 6..7];

fn tokens(rows: &Range<u64>, vocab: u64) -> Vec<u64> {
    rows.clone().map(|i| (i * 3 + 1) % vocab).collect()
}

fn unsplit_logits(
    f: &Fixture,
    vocab: u64,
    orders: Option<&BTreeMap<NodeId, LinearReductionOrder>>,
) -> Vec<Vec<u32>> {
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
            let out = match orders {
                Some(orders) => Interpreter::new()
                    .run_with_linear_orders(
                        &f.graph,
                        &bindings,
                        &mut state,
                        ROOT,
                        &mut cache,
                        &Cancel::never(),
                        orders,
                    )
                    .unwrap(),
                None => Interpreter::new()
                    .run(
                        &f.graph,
                        &bindings,
                        &mut state,
                        ROOT,
                        &mut cache,
                        &Cancel::never(),
                    )
                    .unwrap(),
            };
            bits(&out.logits)
        })
        .collect()
}

fn split_logits(f: &Fixture, lowering: &TensorParallelLowering, vocab: u64) -> Vec<Vec<u32>> {
    let ranks = lowering.ranks.len();
    let stages: Vec<(Join, Vec<StageGraph>)> = lowering
        .stages
        .iter()
        .filter_map(|stage| match stage {
            Stage::Replicated(_) => None,
            Stage::Local { nodes, join } => Some((
                *join,
                lowering
                    .ranks
                    .iter()
                    .map(|part| {
                        stage_graph(
                            &f.graph,
                            &f.weights,
                            nodes.clone(),
                            Some(part),
                            Some(match join {
                                Join::Gather { output } | Join::Reduce { output } => *output,
                            }),
                        )
                    })
                    .collect(),
            )),
        })
        .collect();
    let mut kv: Vec<Vec<(SequenceState, KvCache)>> = Vec::new();
    for stage in &stages {
        if !stage.1[0].graph.state_effects().is_empty() {
            kv.push(
                (0..ranks)
                    .map(|_| {
                        let state = SequenceState::new([StateKind::KvPages]);
                        let cache = KvCache::for_branch(1, &state, ROOT).unwrap();
                        (state, cache)
                    })
                    .collect(),
            );
        }
    }
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
            let mut local_stage = 0;
            let mut state_index = 0;
            for stage in &lowering.stages {
                match stage {
                    Stage::Replicated(nodes) => {
                        let graphs: Vec<StageGraph> = (0..ranks)
                            .map(|_| stage_graph(&f.graph, &f.weights, nodes.clone(), None, None))
                            .collect();
                        for (rank, graph) in graphs.iter().enumerate() {
                            let trace = interpreter
                                .run_stateless(&graph.graph, &bind(graph, &tables[rank]))
                                .unwrap();
                            for (original, id) in &graph.produces {
                                tables[rank]
                                    .insert(*original, trace.node_output(*id).unwrap().clone());
                            }
                        }
                        for table in &tables[1..] {
                            for (original, _) in &graphs[0].produces {
                                assert_eq!(
                                    bits(tables[0][original].as_float().unwrap()),
                                    bits(table[original].as_float().unwrap()),
                                    "replicated value {} differs across ranks",
                                    original.0
                                );
                            }
                        }
                    }
                    Stage::Local { join, .. } => {
                        let (declared_join, graphs) = &stages[local_stage];
                        assert_eq!(*declared_join, *join);
                        let mut outputs = Vec::new();
                        if !graphs[0].graph.state_effects().is_empty() {
                            for (rank, graph) in graphs.iter().enumerate() {
                                let (state, cache) = &mut kv[state_index][rank];
                                state.append_prompt(ROOT, rows.end - rows.start).unwrap();
                                let out = interpreter
                                    .run(
                                        &graph.graph,
                                        &bind(graph, &tables[rank]),
                                        state,
                                        ROOT,
                                        cache,
                                        &Cancel::never(),
                                    )
                                    .unwrap();
                                outputs.push(out.logits);
                            }
                            state_index += 1;
                        } else {
                            for (rank, graph) in graphs.iter().enumerate() {
                                let trace = interpreter
                                    .run_stateless_with_linear_orders(
                                        &graph.graph,
                                        &bind(graph, &tables[rank]),
                                        &graph.linear_orders,
                                    )
                                    .unwrap();
                                outputs.push(trace.output().as_float().unwrap().clone());
                            }
                        }
                        let joined = match join {
                            Join::Gather { .. } => concatenate_columns(&outputs),
                            Join::Reduce { .. } => reduce_rows(&outputs),
                        };
                        let output = match join {
                            Join::Gather { output } | Join::Reduce { output } => *output,
                        };
                        for table in &mut tables {
                            table.insert(output, Value::Float(joined.clone()));
                        }
                        local_stage += 1;
                    }
                }
            }
            bits(tables[0][&f.graph.output()].as_float().unwrap())
        })
        .collect()
}

#[test]
fn dense_tp_is_bit_identical_to_the_unsplit_graph_at_prefill_and_decode() {
    let f = fixture();
    let vocab = 12;
    for ranks in [2, 4] {
        let lowering = lower_tensor_parallel(&f.graph, ranks).unwrap();
        let declared = unsplit_logits(&f, vocab, Some(&lowering.linear_orders));
        assert_eq!(
            split_logits(&f, &lowering, vocab),
            declared,
            "{ranks} ranks"
        );
    }
}

#[test]
fn unsupported_partitions_are_refused() {
    type Expect = fn(&TensorParallelRefused) -> bool;
    let cases: [(&str, Graph, u32, Expect); 6] = [
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
        (
            "a dense MLP with an extra gate consumer",
            mlp_boundary_graph(false),
            2,
            |r| matches!(r, TensorParallelRefused::HeadChain { .. }),
        ),
        (
            "a dense MLP with gate as graph output",
            mlp_boundary_graph(true),
            2,
            |r| matches!(r, TensorParallelRefused::HeadChain { .. }),
        ),
    ];
    for (what, graph, ranks, expected) in cases {
        let refused = lower_tensor_parallel(&graph, ranks).unwrap_err();
        assert!(expected(&refused), "{what}: {refused}");
    }
}

/// A dense GLU whose gate output escapes the local stage through another
/// Linear or the graph boundary. The lowering must refuse the graph instead
/// of leaving that edge in a replicated stage with no rank-local binding.
fn mlp_boundary_graph(output_gate: bool) -> Graph {
    let (hidden, width) = (4u64, 8u64);
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let mut g = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let activation = TensorSpec::new(
        ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
        vec![Dim::symbol(SymbolId(0)), Dim::constant(hidden)],
    );
    let x = g.input("x", activation);
    let weight = |g: &mut GraphBuilder, name: &str, shape: Vec<u64>| {
        let spec = TensorSpec::new(
            ValueRole::Weight(WeightPrecision::new(Precision::Bf16).unwrap()),
            shape.into_iter().map(Dim::constant).collect(),
        );
        g.weight(name, spec).unwrap()
    };
    let gate_weight = weight(&mut g, "gate", vec![width, hidden]);
    let up_weight = weight(&mut g, "up", vec![width, hidden]);
    let down_weight = weight(&mut g, "down", vec![hidden, width]);
    let gate = g
        .node(
            OpParams::Linear {
                in_features: hidden,
                out_features: width,
                bias: false,
            },
            &[x, gate_weight],
        )
        .unwrap();
    let up = g
        .node(
            OpParams::Linear {
                in_features: hidden,
                out_features: width,
                bias: false,
            },
            &[x, up_weight],
        )
        .unwrap();
    let activated = g.node(OpParams::GeGlu { width }, &[gate, up]).unwrap();
    let _down = g
        .node(
            OpParams::Linear {
                in_features: width,
                out_features: hidden,
                bias: false,
            },
            &[activated, down_weight],
        )
        .unwrap();
    let output = if output_gate {
        gate
    } else {
        let extra_weight = weight(&mut g, "extra", vec![hidden, width]);
        g.node(
            OpParams::Linear {
                in_features: width,
                out_features: hidden,
                bias: false,
            },
            &[gate, extra_weight],
        )
        .unwrap()
    };
    g.finish(output, &oracles).unwrap()
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
