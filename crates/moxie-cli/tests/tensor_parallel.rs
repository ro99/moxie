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
    Bindings, Graph, GraphBuilder, IndexEncoding, LinearInputSlice, LinearReductionOrder,
    MlaAttentionDescriptor, NodeId, OpParams, OracleRegistry, RopeLayout, TensorSpec, ValueId,
    ValueRole, Visibility,
};
use moxie_interp::{Interpreter, KvCache};
use moxie_plan::{
    Join, PipelineRefused, RankPart, Stage, TensorParallelLowering, TensorParallelRefused,
    lower_pipeline, lower_tensor_parallel, wavefront,
};
use moxie_state::{MlaLatentDescriptor, ROOT, SequenceState, StateKind};
use moxie_types::{ActivationPrecision, CachePrecision, Dim, Precision, SymbolId, WeightPrecision};

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

fn first_node_of_layer(graph: &Graph, layer: usize) -> usize {
    let suffix = format!(".{layer}");
    graph
        .nodes()
        .iter()
        .position(|node| {
            node.inputs.iter().any(|input| {
                graph.weights().contains(input)
                    && graph
                        .name(*input)
                        .is_some_and(|name| name.ends_with(&suffix))
            })
        })
        .expect("layer has a weight-consuming node")
}

/// One stage as a graph of its own, with original value ids for its boundary.
struct StageGraph {
    graph: Graph,
    bindings: Bindings<Value>,
    reads: Vec<(ValueId, ValueId, Option<LinearInputSlice>)>,
    produces: Vec<(ValueId, ValueId)>,
    linear_orders: BTreeMap<NodeId, LinearReductionOrder>,
    combine_orders: BTreeMap<NodeId, moxie_graph::CombineReductionOrder>,
    expert_ownership: BTreeMap<NodeId, moxie_graph::ExpertOwnership>,
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
            let mut shape = whole.shape().to_vec();
            shape[0] = end - start;
            value = Value::Float(HostTensor::bf16(data, shape).unwrap());
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
        combine_orders: built.combine_orders,
        expert_ownership: built.expert_ownership,
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

fn mla_graph(heads: u64) -> (Graph, Bindings<Value>) {
    let hidden = 8;
    let q_lora = 4;
    let kv_lora = 4;
    let nope = 2;
    let rope = 2;
    let value = 2;
    let descriptor = MlaAttentionDescriptor {
        hidden,
        q_lora_rank: q_lora,
        kv_lora_rank: kv_lora,
        qk_nope_head_dim: nope,
        qk_rope_head_dim: rope,
        v_head_dim: value,
        heads,
        rms_norm_eps: 1e-5,
        rope_base: 10_000.0,
        rope_layout: RopeLayout::Interleaved,
        visibility: Visibility::Causal,
        layer: 0,
        cache_precision: CachePrecision::expect(Precision::Bf16),
    };
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let mut builder = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(88));
    let rows = Dim::symbol(SymbolId(88));
    let hidden_input = builder.input(
        "hidden",
        TensorSpec::new(
            ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
            vec![rows.clone(), Dim::constant(hidden)],
        ),
    );
    let positions = builder.input(
        "positions",
        TensorSpec::new(ValueRole::Index(IndexEncoding::U64), vec![rows]),
    );
    let values = |len: usize, seed: usize| {
        (0..len)
            .map(|i| {
                let raw = ((i * 11 + seed * 7) % 31) as f32 / 31.0 - 0.5;
                moxie_interp::tensor::to_bf16(raw * 0.5)
            })
            .collect::<Vec<_>>()
    };
    let mut weight_bindings = Vec::new();
    let weight_ids: Vec<_> = [
        ("q_a_proj", vec![4, 8], values(4 * 8, 1)),
        ("q_a_layernorm", vec![4], vec![1.0; 4]),
        (
            "q_b_proj",
            vec![heads * 4, 4],
            values(heads as usize * 4 * 4, 2),
        ),
        ("kv_a_proj_with_mqa", vec![6, 8], values(6 * 8, 3)),
        ("kv_a_layernorm", vec![4], vec![1.0; 4]),
        (
            "kv_b_proj",
            vec![heads * 4, 4],
            values(heads as usize * 4 * 4, 4),
        ),
        (
            "o_proj",
            vec![hidden, heads * value],
            values(hidden as usize * heads as usize * value as usize, 5),
        ),
    ]
    .into_iter()
    .map(|(name, shape, data)| {
        let spec = TensorSpec::new(
            ValueRole::Weight(WeightPrecision::expect(Precision::Bf16)),
            shape.iter().copied().map(Dim::constant).collect(),
        );
        let id = builder.weight(name, spec).unwrap();
        weight_bindings.push((
            id,
            data,
            shape.iter().map(|dimension| *dimension as usize).collect(),
        ));
        id
    })
    .collect();
    let [
        q_a_proj,
        q_a_layernorm,
        q_b_proj,
        kv_a_proj,
        kv_a_layernorm,
        kv_b_proj,
        o_proj,
    ] = weight_ids.try_into().unwrap();
    let output = builder
        .node(
            OpParams::MlaAttention { descriptor },
            &[
                hidden_input,
                positions,
                q_a_proj,
                q_a_layernorm,
                q_b_proj,
                kv_a_proj,
                kv_a_layernorm,
                kv_b_proj,
                o_proj,
            ],
        )
        .unwrap();
    let graph = builder.finish(output, &oracles).unwrap();
    let mut bindings = Bindings::new();
    bindings.set(
        hidden_input,
        Value::Float(HostTensor::bf16(values(3 * hidden as usize, 6), vec![3, 8]).unwrap()),
    );
    bindings.set(positions, Value::Index(vec![0, 1, 2]));
    for (id, data, shape) in weight_bindings {
        bindings.set(id, Value::Float(HostTensor::bf16(data, shape).unwrap()));
    }
    (graph, bindings)
}

#[test]
fn mla_heads_split_is_bit_identical_at_prefill_and_decode() {
    let interpreter = Interpreter::new();
    let mla_cache =
        MlaLatentDescriptor::new(4, 2, CachePrecision::expect(Precision::Bf16)).unwrap();
    for ranks in [2, 4] {
        let (graph, mut bindings) = mla_graph(4);
        let lowering = lower_tensor_parallel(&graph, ranks).unwrap();
        assert_eq!(
            lowering.stages,
            vec![Stage::Local {
                nodes: 0..1,
                join: Join::Reduce {
                    output: graph.output(),
                },
            }]
        );

        let mut reference_state = SequenceState::new([StateKind::MlaLatent]);
        reference_state.append_prompt(ROOT, 3).unwrap();
        let mut reference_cache =
            KvCache::for_mla_branch(1, mla_cache, &reference_state, ROOT).unwrap();
        let mut rank_states = Vec::new();
        let mut stages = Vec::new();
        for part in &lowering.ranks {
            let mut state = SequenceState::new([StateKind::MlaLatent]);
            state.append_prompt(ROOT, 3).unwrap();
            let cache = KvCache::for_mla_branch(1, mla_cache, &state, ROOT).unwrap();
            rank_states.push((state, cache));
            stages.push(stage_graph(
                &graph,
                &bindings,
                0..1,
                Some(part),
                Some(graph.output()),
            ));
        }
        let hidden = graph.inputs()[0];
        let positions = graph.inputs()[1];
        for (hidden_data, positions_data, label) in [
            (None, vec![0, 1, 2], "prefill"),
            (
                Some(vec![0.375, -0.25, 0.125, 0.5, -0.125, 0.25, -0.5, 0.375]),
                vec![3],
                "decode",
            ),
        ] {
            if let Some(data) = hidden_data {
                bindings.set(
                    hidden,
                    Value::Float(HostTensor::bf16(data, vec![1, 8]).unwrap()),
                );
            }
            bindings.set(positions, Value::Index(positions_data));
            let table: BTreeMap<ValueId, Value> = graph
                .inputs()
                .iter()
                .map(|input| (*input, bindings.get(*input).unwrap().clone()))
                .collect();
            let reference = interpreter
                .run_with_partition_orders(
                    &graph,
                    &bindings,
                    &mut reference_state,
                    ROOT,
                    &mut reference_cache,
                    &Cancel::never(),
                    &lowering.linear_orders,
                    &lowering.combine_orders,
                    &BTreeMap::new(),
                )
                .unwrap();
            let partials = stages
                .iter()
                .zip(&mut rank_states)
                .map(|(stage, (state, cache))| {
                    interpreter
                        .run_with_partition_orders(
                            &stage.graph,
                            &bind(stage, &table),
                            state,
                            ROOT,
                            cache,
                            &Cancel::never(),
                            &stage.linear_orders,
                            &stage.combine_orders,
                            &stage.expert_ownership,
                        )
                        .unwrap()
                        .logits
                })
                .collect::<Vec<_>>();
            assert_eq!(
                bits(&reduce_rows(&partials)),
                bits(&reference.logits),
                "{ranks} ranks {label}"
            );
        }
    }

    let (graph, _) = mla_graph(4);
    let lowering = lower_tensor_parallel(&graph, 2).unwrap();
    let mut rank_zero = lowering.ranks[0].clone();
    let node = &graph.nodes()[0];
    rank_zero.combine_orders.insert(
        node.id,
        moxie_graph::CombineReductionOrder {
            groups: 2,
            owned: Some(1),
        },
    );
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    assert!(matches!(
        moxie_plan::build_stage_graph(
            &graph,
            Some(&rank_zero),
            0..1,
            Some(graph.output()),
            moxie_oracles::HOST_REFERENCE,
            &oracles,
        ),
        Err(moxie_types::Error::InvalidRequest { field: "stage", .. })
    ));

    let mut truncated = lowering.ranks[0].clone();
    let OpParams::MlaAttention { descriptor } = truncated.params.get_mut(&node.id).unwrap() else {
        unreachable!();
    };
    descriptor.heads = 1;
    let slice = truncated.weight_slices.get_mut(&node.inputs[8]).unwrap();
    slice.width = 2;
    slice.full_width = 4;
    truncated.rows.insert(node.inputs[4], 0..4);
    truncated.rows.insert(node.inputs[7], 0..4);
    assert!(matches!(
        moxie_plan::build_stage_graph(
            &graph,
            Some(&truncated),
            0..1,
            Some(graph.output()),
            moxie_oracles::HOST_REFERENCE,
            &oracles,
        ),
        Err(moxie_types::Error::InvalidRequest { field: "stage", .. })
    ));

    let mut changed_eps = lowering.ranks[0].clone();
    let OpParams::MlaAttention { descriptor } = changed_eps.params.get_mut(&node.id).unwrap()
    else {
        unreachable!();
    };
    descriptor.rms_norm_eps *= 2.0;
    assert!(matches!(
        moxie_plan::build_stage_graph(
            &graph,
            Some(&changed_eps),
            0..1,
            Some(graph.output()),
            moxie_oracles::HOST_REFERENCE,
            &oracles,
        ),
        Err(moxie_types::Error::InvalidRequest { field: "stage", .. })
    ));

    let mut missing_order = lowering.ranks[0].clone();
    missing_order.combine_orders.remove(&node.id);
    assert!(matches!(
        moxie_plan::build_stage_graph(
            &graph,
            Some(&missing_order),
            0..1,
            Some(graph.output()),
            moxie_oracles::HOST_REFERENCE,
            &oracles,
        ),
        Err(moxie_types::Error::InvalidRequest { field: "stage", .. })
    ));

    let (graph, bindings) = mla_graph(3);
    let mut state = SequenceState::new([StateKind::MlaLatent]);
    state.append_prompt(ROOT, 3).unwrap();
    let mut cache = KvCache::for_mla_branch(1, mla_cache, &state, ROOT).unwrap();
    let orders = BTreeMap::from([(
        graph.nodes()[0].id,
        moxie_graph::CombineReductionOrder {
            groups: 2,
            owned: None,
        },
    )]);
    assert!(matches!(
        interpreter.run_with_partition_orders(
            &graph,
            &bindings,
            &mut state,
            ROOT,
            &mut cache,
            &Cancel::never(),
            &BTreeMap::new(),
            &orders,
            &BTreeMap::new(),
        ),
        Err(moxie_types::Error::InvalidRequest {
            field: "mla_order",
            ..
        })
    ));
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

fn unsplit_logits(f: &Fixture, vocab: u64, lowering: &TensorParallelLowering) -> Vec<Vec<u32>> {
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
                .run_with_partition_orders(
                    &f.graph,
                    &bindings,
                    &mut state,
                    ROOT,
                    &mut cache,
                    &Cancel::never(),
                    &lowering.linear_orders,
                    &lowering.combine_orders,
                    &BTreeMap::new(),
                )
                .unwrap();
            bits(&out.logits)
        })
        .collect()
}

fn split_logits(
    f: &Fixture,
    lowering: &TensorParallelLowering,
    vocab: u64,
) -> (Vec<Vec<u32>>, Vec<Vec<Vec<u32>>>) {
    let ranks = lowering.ranks.len();
    let first_route = f
        .graph
        .nodes()
        .iter()
        .find(|node| matches!(node.params, OpParams::Route { .. }))
        .map(|node| node.output);
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
    let mut routes_by_step = Vec::new();
    let logits = STEPS
        .iter()
        .map(|rows| {
            let mut route_rows = Vec::new();
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
                                let value = trace.node_output(*id).unwrap().clone();
                                tables[rank].insert(*original, value);
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
                                    .run_stateless_with_partition_orders(
                                        &graph.graph,
                                        &bind(graph, &tables[rank]),
                                        &graph.linear_orders,
                                        &graph.combine_orders,
                                        &graph.expert_ownership,
                                    )
                                    .unwrap();
                                if rank == 0 {
                                    for (original, id) in &graph.produces {
                                        if Some(*original) == first_route {
                                            let route =
                                                trace.node_output(*id).unwrap().as_route().unwrap();
                                            route_rows.extend((0..route.rows()).map(|row| {
                                                route.row_experts(row).unwrap().to_vec()
                                            }));
                                        }
                                    }
                                }
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
            routes_by_step.push(route_rows);
            bits(tables[0][&f.graph.output()].as_float().unwrap())
        })
        .collect();
    (logits, routes_by_step)
}

#[test]
fn dense_tp_is_bit_identical_to_the_unsplit_graph_at_prefill_and_decode() {
    let f = fixture();
    let vocab = 12;
    for ranks in [2, 4] {
        let lowering = lower_tensor_parallel(&f.graph, ranks).unwrap();
        let declared = unsplit_logits(&f, vocab, &lowering);
        let (actual, _) = split_logits(&f, &lowering, vocab);
        assert_eq!(actual, declared, "{ranks} ranks");
    }
}

#[test]
fn routed_expert_owners_are_bit_identical_at_prefill_and_decode() {
    let mut config = gemma::Shape::C.config();
    config.moe.as_mut().unwrap().experts = 8;
    config.vocab = 12;
    let mut f = gemma::build_with_config(config).unwrap();
    let weight = |name: &str| {
        *f.graph
            .weights()
            .iter()
            .find(|id| {
                f.graph
                    .name(**id)
                    .is_some_and(|label| label.starts_with(&format!("{name}.")))
            })
            .unwrap()
    };
    let projection_id = weight("router_proj");
    let width = f
        .weights
        .get(projection_id)
        .unwrap()
        .as_float()
        .unwrap()
        .cols();
    let mut projection = vec![0.0; 8 * width];
    projection[0] = 1.0;
    projection[4 * width] = -1.0;
    f.weights.set(
        projection_id,
        Value::Float(HostTensor::bf16(projection, vec![8, width]).unwrap()),
    );
    let scale_id = weight("router_scale");
    f.weights.set(
        scale_id,
        Value::Float(HostTensor::bf16(vec![1.0; width], vec![width]).unwrap()),
    );
    let vocab = 12;

    for ranks in [2, 4] {
        let lowering = lower_tensor_parallel(&f.graph, ranks).unwrap();
        let declared = unsplit_logits(&f, vocab, &lowering);
        let (actual, routes_by_step) = split_logits(&f, &lowering, vocab);
        assert_eq!(actual, declared, "{ranks} ranks");

        let prefill_routes = &routes_by_step[0];
        let group_width = 8 / ranks;
        let owners = |row: &[u32]| {
            row.iter()
                .map(|expert| *expert / group_width)
                .collect::<Vec<_>>()
        };
        let same_owner = prefill_routes
            .iter()
            .enumerate()
            .find(|(_, row)| owners(row).windows(2).all(|pair| pair[0] == pair[1]));
        assert!(
            same_owner.is_some(),
            "{ranks} ranks had no duplicate destination in prefill routes {prefill_routes:?}"
        );
        let different_owners = prefill_routes
            .iter()
            .enumerate()
            .find(|(_, row)| owners(row).windows(2).any(|pair| pair[0] != pair[1]));
        assert!(
            different_owners.is_some(),
            "{ranks} ranks had no cross-owner route in prefill routes {prefill_routes:?}"
        );
        if ranks == 4 {
            let selected: Vec<u32> = prefill_routes.iter().flat_map(|row| owners(row)).collect();
            assert!(
                (0..ranks).any(|owner| !selected.contains(&owner)),
                "every owner received a prefill route: {prefill_routes:?}"
            );
        }
    }
}

#[test]
fn pipeline_stages_are_bit_identical_with_microbatched_prefill() {
    let dense = fixture();
    let mut config = gemma::Shape::C.config();
    config.moe.as_mut().unwrap().experts = 8;
    config.vocab = 12;
    let routed = gemma::build_with_config(config).unwrap();
    let interpreter = Interpreter::new();

    for (label, f) in [("dense", &dense), ("routed", &routed)] {
        let cuts = [
            first_node_of_layer(&f.graph, 1),
            first_node_of_layer(&f.graph, 4),
        ];
        let lowering = lower_pipeline(&f.graph, &cuts).unwrap_or_else(|refused| {
            panic!("{label} Gemma layer boundary violates the one-handoff contract: {refused:?}")
        });
        let stages: Vec<_> = lowering
            .stages()
            .iter()
            .cloned()
            .map(|nodes| stage_graph(&f.graph, &f.weights, nodes, None, None))
            .collect();

        let mut reference_state = SequenceState::new([StateKind::KvPages]);
        let mut reference_cache =
            KvCache::for_branch(f.graph.attention_layers().len(), &reference_state, ROOT).unwrap();
        let reference: Vec<_> = STEPS
            .iter()
            .map(|rows| {
                reference_state
                    .append_prompt(ROOT, rows.end - rows.start)
                    .unwrap();
                let mut bindings = f.weights.clone();
                bindings.set(f.tokens, Value::Index(tokens(rows, 12)));
                bindings.set(f.positions, Value::Index(rows.clone().collect()));
                let output = interpreter
                    .run(
                        &f.graph,
                        &bindings,
                        &mut reference_state,
                        ROOT,
                        &mut reference_cache,
                        &Cancel::never(),
                    )
                    .unwrap();
                bits(&output.logits)
            })
            .collect();

        let mut states: Vec<_> = stages
            .iter()
            .map(|_| SequenceState::new([StateKind::KvPages]))
            .collect();
        let mut caches: Vec<_> = stages
            .iter()
            .enumerate()
            .map(|(index, stage)| {
                KvCache::for_branch(stage.graph.attention_layers().len(), &states[index], ROOT)
                    .unwrap()
            })
            .collect();
        let microbatches: [Range<u64>; 2] = [0..2, 2..5];
        let mut tables: Vec<_> = microbatches
            .iter()
            .map(|rows| {
                let mut table = BTreeMap::new();
                table.insert(f.tokens, Value::Index(tokens(rows, 12)));
                table.insert(f.positions, Value::Index(rows.clone().collect()));
                table
            })
            .collect();
        let mut actual = Vec::with_capacity(STEPS.len());
        let mut prefill = Vec::new();
        for (stage_index, microbatch) in wavefront(3, 2) {
            let rows = &microbatches[microbatch];
            states[stage_index]
                .append_prompt(ROOT, rows.end - rows.start)
                .unwrap();
            let output = interpreter
                .run(
                    &stages[stage_index].graph,
                    &bind(&stages[stage_index], &tables[microbatch]),
                    &mut states[stage_index],
                    ROOT,
                    &mut caches[stage_index],
                    &Cancel::never(),
                )
                .unwrap();
            if stage_index + 1 < stages.len() {
                tables[microbatch].insert(
                    lowering.handoffs()[stage_index],
                    Value::Float(output.logits),
                );
            } else {
                prefill.extend(bits(&output.logits));
            }
        }
        actual.push(prefill);

        for rows in STEPS.iter().skip(1) {
            let mut table = BTreeMap::new();
            table.insert(f.tokens, Value::Index(tokens(rows, 12)));
            table.insert(f.positions, Value::Index(rows.clone().collect()));
            let mut logits = Vec::new();
            for stage_index in 0..stages.len() {
                states[stage_index]
                    .append_prompt(ROOT, rows.end - rows.start)
                    .unwrap();
                let output = interpreter
                    .run(
                        &stages[stage_index].graph,
                        &bind(&stages[stage_index], &table),
                        &mut states[stage_index],
                        ROOT,
                        &mut caches[stage_index],
                        &Cancel::never(),
                    )
                    .unwrap();
                if stage_index + 1 < stages.len() {
                    table.insert(
                        lowering.handoffs()[stage_index],
                        Value::Float(output.logits),
                    );
                } else {
                    logits = bits(&output.logits);
                }
            }
            actual.push(logits);
        }
        assert_eq!(actual, reference, "{label} pipeline");
    }
}

#[test]
fn unsupported_partitions_are_refused() {
    type Expect = fn(&TensorParallelRefused) -> bool;
    let cases: [(&str, Graph, u32, Expect); 9] = [
        (
            "4 query heads over 3 ranks",
            gemma::build(gemma::Shape::A).unwrap().graph,
            3,
            |r| matches!(r, TensorParallelRefused::Heads { .. }),
        ),
        ("2 MLA heads over 4 ranks", mla_graph(2).0, 4, |r| {
            matches!(
                r,
                TensorParallelRefused::Heads {
                    heads: 2,
                    ranks: 4,
                    ..
                }
            )
        }),
        (
            "3 global kv heads over 2 ranks",
            gemma::build(gemma::Shape::B).unwrap().graph,
            2,
            |r| matches!(r, TensorParallelRefused::KvHeads { .. }),
        ),
        (
            "five experts over two ranks",
            gemma::build(gemma::Shape::C).unwrap().graph,
            2,
            |r| {
                matches!(
                    r,
                    TensorParallelRefused::Dimension {
                        axis: "experts",
                        ..
                    }
                )
            },
        ),
        (
            "expert slots escaping their local stage",
            routed_graph(4, 1.0, true),
            2,
            |r| matches!(r, TensorParallelRefused::HeadChain { .. }),
        ),
        (
            "a non-unit routed combine scale",
            routed_graph(4, 2.0, false),
            2,
            |r| matches!(r, TensorParallelRefused::ScaledCombine { .. }),
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

    let dense = fixture();
    for cuts in [&[][..], &[0][..], &[2, 1][..]] {
        assert!(matches!(
            lower_pipeline(&dense.graph, cuts),
            Err(PipelineRefused::Cuts)
        ));
    }
    let mut config = gemma::Shape::C.config();
    config.moe.as_mut().unwrap().experts = 8;
    config.vocab = 12;
    let routed = gemma::build_with_config(config).unwrap();
    let route = routed
        .graph
        .nodes()
        .iter()
        .enumerate()
        .find(|(_, node)| matches!(node.params, OpParams::Route { .. }))
        .unwrap();
    assert!(matches!(
        lower_pipeline(&routed.graph, &[route.0 + 1]),
        Err(PipelineRefused::NonActivation { value }) if value == route.1.output
    ));
    let attention = dense
        .graph
        .nodes()
        .iter()
        .position(|node| matches!(node.params, OpParams::Attention { .. }))
        .unwrap();
    assert!(matches!(
        lower_pipeline(&dense.graph, &[attention]),
        Err(PipelineRefused::Handoffs { boundary: 0, values }) if values.len() > 1
    ));
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

fn routed_graph(experts: u64, output_scale: f32, slots_escape: bool) -> Graph {
    let (hidden, intermediate, top_k) = (4u64, 2u64, 2u64);
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let mut g = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
    let x = g.input(
        "x",
        TensorSpec::new(
            ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16)),
            vec![Dim::symbol(SymbolId(0)), Dim::constant(hidden)],
        ),
    );
    let weight = |g: &mut GraphBuilder, name: &str, shape: Vec<u64>| {
        let spec = TensorSpec::new(
            ValueRole::Weight(WeightPrecision::new(Precision::Bf16).unwrap()),
            shape.into_iter().map(Dim::constant).collect(),
        );
        g.weight(name, spec).unwrap()
    };
    let router = weight(&mut g, "router", vec![experts, hidden]);
    let gate_up = weight(&mut g, "gate_up", vec![experts, 2 * intermediate, hidden]);
    let down = weight(&mut g, "down", vec![experts, hidden, intermediate]);
    let route = g
        .node(
            OpParams::Route {
                hidden,
                experts,
                top_k,
                input: moxie_graph::RouterInput::Raw,
                score: moxie_graph::RouteScore::Softmax,
                per_expert_scale: false,
                selection_bias: false,
                coefficient: moxie_graph::RouteCoefficient::Fp32,
            },
            &[x, router],
        )
        .unwrap();
    let slots = g
        .node(
            OpParams::ExpertMlp {
                hidden,
                intermediate,
                experts,
                top_k,
                activation: moxie_graph::ExpertActivation::GeGlu,
            },
            &[x, route, gate_up, down],
        )
        .unwrap();
    let combined = g
        .node(
            OpParams::Combine {
                hidden,
                top_k,
                order: moxie_graph::CombineOrder::AscendingExpertId,
                output_scale,
            },
            &[route, slots],
        )
        .unwrap();
    g.finish(if slots_escape { slots } else { combined }, &oracles)
        .unwrap()
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
