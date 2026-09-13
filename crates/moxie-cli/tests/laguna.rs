//! Task 0022: Laguna's metadata and its routed block, and what neither
//! establishes.
//!
//! **Nothing here reads a Laguna tensor.** One test reads the artifact's
//! `config.json` -- a metadata file, not a payload -- to check that the
//! geometry `moxie_models::laguna::ARTIFACT` declares is the geometry the
//! artifact declares, and prints `SKIPPED` with a reason when the artifact is
//! not on the machine. The weights every other test uses are a deterministic
//! pattern this file invents, and the logits they produce are not model output.

use moxie_engine::{HostTensor, Value};
use moxie_graph::{
    Bindings, CombineOrder, ExpertActivation, GraphBuilder, OpParams, OracleRegistry,
    RouteCoefficient, RouteScore, RouterInput, TensorSpec, ValueId, ValueRole,
};
use moxie_interp::Interpreter;
use moxie_models::laguna::{
    ARTIFACT, AttentionGate, BlockConfig, Composition, Gap, RopeKind, RoutedBlocks,
};
use moxie_types::{Dim, Precision, SymbolId, WeightPrecision};

const ARTIFACT_ROOT: &str = "/fast/models/cyankiwi/Laguna-S-2.1-AWQ-INT4";

fn oracles() -> OracleRegistry {
    let mut r = OracleRegistry::new();
    moxie_oracles::register(&mut r).expect("oracles");
    r
}

/// Bind every declared role with a deterministic pattern.
fn bind(composed: &Composition, tokens: &[u64]) -> Bindings<Value> {
    let mut weights = Bindings::new();
    for (index, bound) in composed.weights.iter().enumerate() {
        let spec = composed.graph.spec(bound.value).expect("role has a value");
        let extent: Vec<usize> = spec
            .shape
            .iter()
            .map(|d| match d {
                Dim::Const(n) => *n as usize,
                other => panic!("weight {} has extent {other:?}", bound.role.name),
            })
            .collect();
        let count: usize = extent.iter().product();
        let data: Vec<f32> = (0..count)
            .map(|i| {
                let mixed = (i * 13 + index * 7 + 3) % 31;
                (mixed as f32 - 15.0) / 32.0
            })
            .collect();
        weights.set(
            bound.value,
            Value::Float(HostTensor::bf16(data, extent).expect("bf16")),
        );
    }
    weights.set(composed.tokens, Value::Index(tokens.to_vec()));
    weights
}

fn logits_of(config: BlockConfig, tokens: &[u64]) -> Vec<f32> {
    let model = RoutedBlocks::reduced(config, "synthetic-routed-block").expect("reduced");
    let registry = oracles();
    moxie_model_api::admit(&model, &registry).expect("admitted");
    let composed = model.compose(&registry, SymbolId(0)).expect("composed");
    let bindings = bind(&composed, tokens);
    let trace = Interpreter::new()
        .run_stateless(&composed.graph, &bindings)
        .expect("stateless run");
    trace.output().as_float().expect("float").data().to_vec()
}

#[test]
fn the_routed_block_composes_and_runs() {
    let logits = logits_of(BlockConfig::reduced(), &[0, 1, 2, 3]);
    let config = BlockConfig::reduced();
    assert_eq!(logits.len(), 4 * config.vocab as usize);
    assert!(logits.iter().all(|v| v.is_finite()));
}

#[test]
fn the_block_is_row_independent() {
    // A routed feed-forward block has no sequence state: every row is routed
    // and computed on its own. So evaluating four rows together must equal
    // evaluating them one at a time -- the stateless form of the
    // whole-versus-chunked parity every other consumer owes.
    //
    // It is worth checking rather than assuming: `ExpertMlp` is the operation a
    // grouped kernel batches rows through, and a grouping that leaked one row's
    // intermediate into another's would still produce well-shaped logits.
    let config = BlockConfig::reduced();
    let whole = logits_of(config.clone(), &[0, 1, 2, 3]);
    let vocab = config.vocab as usize;
    for (row, token) in [0u64, 1, 2, 3].iter().enumerate() {
        let alone = logits_of(config.clone(), &[*token]);
        assert_eq!(
            &whole[row * vocab..(row + 1) * vocab],
            &alone[..],
            "row {row} depends on the rows beside it"
        );
    }
}

#[test]
fn the_routed_scaling_factor_is_load_bearing() {
    // `moe_routed_scaling_factor` is 2.5 in the artifact, and it reaches the
    // graph as `Combine`'s `output_scale`. If it did not, this fixture would
    // not notice the difference between 2.5 and 1.0.
    let base = BlockConfig::reduced();
    let mut unscaled = base.clone();
    unscaled.moe.routed_scaling_factor = 1.0;
    assert_ne!(
        logits_of(base, &[0, 1, 2]),
        logits_of(unscaled, &[0, 1, 2]),
        "the routed scaling factor reached nothing"
    );
}

#[test]
fn the_selection_bias_is_load_bearing() {
    // The bias is a bound tensor, so the way to show it matters is to bind it
    // to zero and require a different answer. A bias that changed nothing would
    // mean the router was selecting on the unbiased score.
    let config = BlockConfig::reduced();
    let model = RoutedBlocks::reduced(config.clone(), "synthetic").expect("reduced");
    let registry = oracles();
    let composed = model.compose(&registry, SymbolId(0)).expect("composed");
    let tokens = [0u64, 1, 2, 3, 4, 5];

    let base = {
        let bindings = bind(&composed, &tokens);
        let trace = Interpreter::new()
            .run_stateless(&composed.graph, &bindings)
            .expect("run");
        trace.output().as_float().expect("float").data().to_vec()
    };
    let zeroed = {
        let mut bindings = bind(&composed, &tokens);
        for bound in &composed.weights {
            if bound.role.name == "router_selection_bias" {
                let spec = composed.graph.spec(bound.value).expect("spec");
                let count: usize = spec
                    .shape
                    .iter()
                    .map(|d| match d {
                        Dim::Const(n) => *n as usize,
                        other => panic!("{other:?}"),
                    })
                    .product();
                bindings.set(
                    bound.value,
                    Value::Float(HostTensor::bf16(vec![0.0; count], vec![count]).expect("bf16")),
                );
            }
        }
        let trace = Interpreter::new()
            .run_stateless(&composed.graph, &bindings)
            .expect("run");
        trace.output().as_float().expect("float").data().to_vec()
    };
    assert_ne!(base, zeroed, "the selection bias reached nothing");
}

#[test]
fn the_router_and_the_shared_expert_read_the_same_normalization() {
    // The opposite of Gemma 4, whose experts read their own
    // `pre_feedforward_layernorm_2` and whose router reads the un-normalized
    // residual. Laguna's block normalizes once and hands the result to both, so
    // there is exactly one `RmsNorm` per block and the `Route` node's producer
    // is it.
    let config = BlockConfig::reduced();
    let model = RoutedBlocks::reduced(config.clone(), "synthetic").expect("reduced");
    let composed = model.compose(&oracles(), SymbolId(0)).expect("composed");
    let nodes = composed.graph.nodes();

    let route = nodes
        .iter()
        .find(|n| matches!(n.params, OpParams::Route { .. }))
        .expect("a route node");
    let experts = nodes
        .iter()
        .find(|n| matches!(n.params, OpParams::ExpertMlp { .. }))
        .expect("an expert node");
    // The shared expert's gate projection: the first `Linear` of the block.
    let shared_gate = nodes
        .iter()
        .find(|n| matches!(n.params, OpParams::Linear { .. }))
        .expect("a linear node");

    assert_eq!(
        route.inputs[0], experts.inputs[0],
        "the router and the experts must read the same tensor"
    );
    assert_eq!(
        route.inputs[0], shared_gate.inputs[0],
        "the shared expert must read the same tensor as the router"
    );
    let producer = nodes
        .iter()
        .find(|n| n.output == route.inputs[0])
        .expect("the router's input has a producer");
    assert!(
        matches!(producer.params, OpParams::RmsNorm { .. }),
        "the router reads {:?}, not a normalization",
        producer.params
    );
}

#[test]
fn the_two_expert_vectors_are_told_apart_by_an_oracle_not_only_by_each_other() {
    // `per_expert_scale` and `selection_bias` are both `[experts]`, so a graph
    // that bound them the wrong way round passes every shape check.
    // `OpParams::route_operands` is the one statement of the order.
    //
    // The first version of this test only required the two bindings to give
    // *different* answers. A mutation battery showed that was not enough:
    // reversing `route_operands` relabels both the validation and the
    // interpretation consistently, so a swapped order still produced two
    // different answers and survived. The check has to be against an
    // **independent** statement of what each operand means -- the oracle's --
    // which is the same lesson as task 0021's swapped kernel symbol: a gate
    // only fires on inputs something hands it, and an assertion that compares a
    // thing to itself is not one.
    let hidden = 8u64;
    let experts = 5u64;
    let top_k = 3u64;
    let intermediate = 4u64;
    let rows_count = 4usize;

    // Every fixture value, already rounded to BF16 exactly as a bound weight
    // is, so the oracle below reads the same numbers the interpreter does.
    let pattern = |count: usize, off: u64| -> Vec<f32> {
        let raw: Vec<f32> = (0..count)
            .map(|i| (((i as u64 * 17 + off) % 29) as f32 - 14.0) / 32.0)
            .collect();
        HostTensor::bf16(raw, vec![count])
            .expect("bf16")
            .data()
            .to_vec()
    };
    let proj_values = pattern((experts * hidden) as usize, 3);
    let per_expert_values = pattern(experts as usize, 5);
    let bias_values = pattern(experts as usize, 19);
    let gate_up_values = pattern((experts * 2 * intermediate * hidden) as usize, 7);
    let down_values = pattern((experts * hidden * intermediate) as usize, 11);
    let row_values: Vec<f32> = {
        let raw: Vec<f32> = (0..rows_count * hidden as usize)
            .map(|i| ((i * 5 % 23) as f32 - 11.0) / 16.0)
            .collect();
        HostTensor::bf16(raw, vec![rows_count * hidden as usize])
            .expect("bf16")
            .data()
            .to_vec()
    };

    let run = |swapped: bool| -> Vec<f32> {
        let mut g = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(0));
        let rows = g.input(
            "rows",
            TensorSpec::new(
                ValueRole::Activation(moxie_types::ActivationPrecision::expect(Precision::Bf16)),
                vec![Dim::symbol(SymbolId(0)), Dim::constant(hidden)],
            ),
        );
        let mut weights = Bindings::new();
        let mut param = |g: &mut GraphBuilder,
                         name: &str,
                         shape: Vec<u64>,
                         data: &[f32]|
         -> ValueId {
            let id = g
                .weight(
                    name,
                    TensorSpec::new(
                        ValueRole::Weight(WeightPrecision::new(Precision::Bf16).unwrap()),
                        shape.iter().map(|d| Dim::constant(*d)).collect(),
                    ),
                )
                .expect("weight");
            weights.set(
                id,
                Value::Float(
                    HostTensor::bf16(data.to_vec(), shape.iter().map(|d| *d as usize).collect())
                        .expect("bf16"),
                ),
            );
            id
        };
        let proj = param(
            &mut g,
            "router projection",
            vec![experts, hidden],
            &proj_values,
        );
        let per_expert = param(
            &mut g,
            "per-expert scale",
            vec![experts],
            &per_expert_values,
        );
        let bias = param(&mut g, "selection bias", vec![experts], &bias_values);
        let gate_up = param(
            &mut g,
            "fused gate/up",
            vec![experts, 2 * intermediate, hidden],
            &gate_up_values,
        );
        let down = param(
            &mut g,
            "fused down",
            vec![experts, hidden, intermediate],
            &down_values,
        );

        let operands = if swapped {
            [rows, proj, bias, per_expert]
        } else {
            [rows, proj, per_expert, bias]
        };
        let route = g
            .node(
                OpParams::Route {
                    hidden,
                    experts,
                    top_k,
                    input: RouterInput::Raw,
                    score: RouteScore::Sigmoid,
                    per_expert_scale: true,
                    selection_bias: true,
                    coefficient: RouteCoefficient::Bf16,
                },
                &operands,
            )
            .expect("route");
        let slots = g
            .node(
                OpParams::ExpertMlp {
                    hidden,
                    intermediate,
                    experts,
                    top_k,
                    activation: ExpertActivation::SwiGlu,
                },
                &[rows, route, gate_up, down],
            )
            .expect("experts");
        let combined = g
            .node(
                OpParams::Combine {
                    hidden,
                    top_k,
                    order: CombineOrder::AscendingExpertId,
                    output_scale: 2.5,
                },
                &[route, slots],
            )
            .expect("combine");
        let graph = g.finish(combined, &oracles()).expect("finish");
        weights.set(
            rows,
            Value::Float(
                HostTensor::bf16(row_values.clone(), vec![rows_count, hidden as usize])
                    .expect("bf16"),
            ),
        );
        let trace = Interpreter::new()
            .run_stateless(&graph, &weights)
            .expect("run");
        trace.output().as_float().expect("float").data().to_vec()
    };

    // The oracle, composed from the accepted stages with each operand in the
    // role its name says.
    let spec = moxie_oracles::route::RouterSpec {
        experts: experts as usize,
        top_k: top_k as usize,
        input: RouterInput::Raw,
        score: RouteScore::Sigmoid,
        coefficient: RouteCoefficient::Bf16,
    };
    let expert_spec = moxie_oracles::route::ExpertSpec {
        experts: experts as usize,
        hidden: hidden as usize,
        intermediate: intermediate as usize,
        activation: ExpertActivation::SwiGlu,
    };
    let mut want = Vec::new();
    for r in 0..rows_count {
        let x = &row_values[r * hidden as usize..(r + 1) * hidden as usize];
        let route = moxie_oracles::route::router_route_row(
            x,
            None,
            &proj_values,
            Some(&per_expert_values),
            Some(&bias_values),
            spec,
        )
        .expect("route");
        let mut slots = Vec::new();
        for e in &route.experts {
            let y =
                moxie_oracles::route::expert_row(x, &gate_up_values, &down_values, *e, expert_spec)
                    .expect("expert");
            // The node's own BF16 output boundary, which the interpreter
            // applies to the slot tensor.
            slots.extend(y.iter().map(|v| moxie_oracles::bf16_round(*v)));
        }
        let row = moxie_oracles::route::combine_row(
            &route.experts,
            &route.weights,
            &slots,
            hidden as usize,
            CombineOrder::AscendingExpertId,
            2.5,
        )
        .expect("combine");
        want.extend(row.iter().map(|v| moxie_oracles::bf16_round(*v)));
    }

    let declared = run(false);
    assert_eq!(
        declared, want,
        "the declared operand order does not mean what the oracle means by it"
    );
    assert_ne!(
        declared,
        run(true),
        "the two `[experts]` operands are interchangeable, so nothing distinguishes them"
    );
}

#[test]
fn a_nonzero_router_logit_softcap_is_refused() {
    // `Route` has no softcap parameter, because no inspected artifact declares
    // one and a branch nothing exercises is a stub. A configuration that needs
    // it stops here, where the gap is visible, rather than being composed as
    // though the cap were not there.
    let mut config = BlockConfig::reduced();
    config.moe.router_logit_softcap = 30.0;
    let err = RoutedBlocks::reduced(config, "synthetic").expect_err("must refuse");
    let text = format!("{err}");
    assert!(
        text.contains("softcap") || text.contains("softcapping"),
        "{text}"
    );
    assert_eq!(ARTIFACT.moe.router_logit_softcap, 0.0);
}

#[test]
fn the_artifact_declares_the_two_gaps_that_block_its_tower() {
    let gaps = ARTIFACT.tower_gaps();
    assert!(gaps.contains(&Gap::AttentionOutputGate), "{gaps:?}");
    assert!(gaps.contains(&Gap::YarnRope), "{gaps:?}");
    for gap in &gaps {
        assert!(!gap.detail().is_empty());
    }

    // And the same function reports none for a geometry without them, which is
    // what keeps it a computation rather than a constant.
    let mut composable = ARTIFACT;
    composable.attention_gate = AttentionGate::None;
    composable.full_rope = ARTIFACT.sliding_rope;
    assert!(composable.tower_gaps().is_empty());
}

#[test]
fn the_reduction_says_what_the_block_is_not() {
    let model = RoutedBlocks::reduced(BlockConfig::reduced(), "synthetic").expect("reduced");
    let r = model.reduction();
    assert!(r.synthetic_weights);
    assert!(r.no_attention);
    assert!(r.routed_block_only);
}

/// The expert inventory, against the artifact's own shard headers.
///
/// A review found the record's figure wrong by **6.87 GB**: it multiplied the
/// packed per-layer cost by all 47 routed layers, in a document that had
/// already recorded that two of them keep BF16 experts. The arithmetic is
/// executable now, and this is what makes it so — it reads the index and the
/// headers, sums the expert tensors, and compares against what `ARTIFACT`
/// declares. Metadata only: `data_offsets`, never a payload.
#[test]
fn the_expert_inventory_matches_the_artifact_headers() {
    let index_path = format!("{ARTIFACT_ROOT}/model.safetensors.index.json");
    let Ok(text) = std::fs::read_to_string(&index_path) else {
        eprintln!("SKIPPED: {index_path} is not present on this machine");
        return;
    };
    let index: serde_json::Value = serde_json::from_str(&text).expect("index parses");
    let map = index["weight_map"].as_object().expect("weight_map");

    // One header read per shard, not per tensor.
    let mut headers: std::collections::BTreeMap<String, serde_json::Value> =
        std::collections::BTreeMap::new();
    for shard in map.values() {
        let shard = shard.as_str().expect("shard name");
        if headers.contains_key(shard) {
            continue;
        }
        let mut f = std::fs::File::open(format!("{ARTIFACT_ROOT}/{shard}")).expect("shard");
        let mut len = [0u8; 8];
        std::io::Read::read_exact(&mut f, &mut len).expect("header length");
        let mut buf = vec![0u8; u64::from_le_bytes(len) as usize];
        std::io::Read::read_exact(&mut f, &mut buf).expect("header");
        headers.insert(
            shard.to_string(),
            serde_json::from_slice(&buf).expect("header parses"),
        );
    }

    let mut measured: std::collections::BTreeMap<u32, u64> = std::collections::BTreeMap::new();
    for (name, shard) in map {
        if !name.contains(".mlp.experts.") || name.contains("e_score_correction_bias") {
            continue;
        }
        let layer: u32 = name
            .split('.')
            .nth(2)
            .expect("layer")
            .parse()
            .expect("index");
        let entry = &headers[shard.as_str().expect("shard")][name]["data_offsets"];
        let bytes = entry[1].as_u64().expect("end") - entry[0].as_u64().expect("start");
        *measured.entry(layer).or_default() += bytes;
    }

    for (layer, bytes) in &measured {
        assert_eq!(
            ARTIFACT.layer_expert_bytes(*layer),
            Some(*bytes),
            "layer {layer}"
        );
    }
    let routed: Vec<u32> = (0..ARTIFACT.layers)
        .filter(|l| ARTIFACT.layer_expert_bytes(*l).is_some())
        .collect();
    assert_eq!(
        routed.len(),
        measured.len(),
        "the declared routed layers are not the ones with expert tensors"
    );
    let total: u64 = measured.values().sum();
    assert_eq!(ARTIFACT.expert_bytes_total(), total);
    assert_eq!(total, 72_515_874_816);

    // And the fraction the record quotes, to a tenth of a percent.
    let fraction = total as f64 / ARTIFACT.artifact_bytes as f64;
    assert!((fraction - 0.9441).abs() < 0.0005, "{fraction}");

    // The exception is load-bearing: assuming every routed layer cost the
    // packed figure is the error the review found, and it is this large.
    let uniform = 47 * ARTIFACT.moe.experts * ARTIFACT.stored_expert_bytes;
    assert_eq!(total - uniform, 6_870_245_376);
}

#[test]
fn one_expert_of_the_artifact_costs_what_the_record_says() {
    // The figure a BF16 fixture at this geometry demands, and the one a
    // restricted budget is stated as a ratio of. The artifact's own per-expert
    // cost is smaller because its experts are INT4; that number is in
    // `docs/models/laguna.md` and is not this one.
    assert_eq!(ARTIFACT.expert_bf16_bytes(), 18_874_368);
    assert_eq!(ARTIFACT.stored_expert_bytes, 5_455_920);
    assert_eq!(ARTIFACT.moe.experts, 256);
    assert_eq!(ARTIFACT.moe.top_k, 10);
    // The two routed layers the quantizer left alone cost the BF16 figure.
    assert_eq!(ARTIFACT.layer_expert_bytes(1), Some(1_396_715_520));
    assert_eq!(ARTIFACT.layer_expert_bytes(46), Some(4_831_838_208));
    assert_eq!(ARTIFACT.layer_expert_bytes(0), None, "layer 0 is dense");
}

#[test]
fn the_layer_predicates_match_the_declared_arrays() {
    // `layer_types` is full-attention at 0, 4, 8, ... and `mlp_only_layers` is
    // `[0]`. Both are predicates here rather than 48-entry arrays, so this is
    // the test that says the predicates reproduce the arrays.
    let full: Vec<u32> = (0..ARTIFACT.layers)
        .filter(|l| ARTIFACT.full_attention_layer(*l))
        .collect();
    assert_eq!(full.len(), 12);
    assert_eq!(full.first(), Some(&0));
    assert_eq!(full.last(), Some(&44));

    let dense: Vec<u32> = (0..ARTIFACT.layers)
        .filter(|l| ARTIFACT.dense_layer(*l))
        .collect();
    assert_eq!(dense, vec![0]);
}

/// `ARTIFACT` against the artifact's own `config.json`, field by field.
///
/// Reads one metadata file. No tensor payload is opened, and nothing is
/// written. Prints `SKIPPED` with a reason when the artifact is not present.
#[test]
fn the_declared_geometry_matches_the_artifact_config() {
    let path = format!("{ARTIFACT_ROOT}/config.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("SKIPPED: {path} is not present on this machine");
        return;
    };
    let c: serde_json::Value = serde_json::from_str(&text).expect("config.json parses");
    let u = |k: &str| -> u64 { c[k].as_u64().unwrap_or_else(|| panic!("{k} missing")) };
    let f = |k: &str| -> f64 { c[k].as_f64().unwrap_or_else(|| panic!("{k} missing")) };

    assert_eq!(u("hidden_size"), ARTIFACT.hidden);
    assert_eq!(u("num_hidden_layers"), u64::from(ARTIFACT.layers));
    assert_eq!(u("head_dim"), ARTIFACT.head_dim);
    assert_eq!(u("num_key_value_heads"), ARTIFACT.kv_heads);
    assert_eq!(u("intermediate_size"), ARTIFACT.intermediate);
    assert_eq!(u("vocab_size"), ARTIFACT.vocab);
    assert_eq!(u("sliding_window"), ARTIFACT.sliding_window);
    assert_eq!(u("max_position_embeddings"), ARTIFACT.max_trained_position);
    assert_eq!(u("num_experts"), ARTIFACT.moe.experts);
    assert_eq!(u("num_experts_per_tok"), ARTIFACT.moe.top_k);
    assert_eq!(u("moe_intermediate_size"), ARTIFACT.moe.moe_intermediate);
    assert_eq!(
        u("shared_expert_intermediate_size"),
        ARTIFACT.moe.shared_intermediate
    );
    assert_eq!(
        f("moe_routed_scaling_factor") as f32,
        ARTIFACT.moe.routed_scaling_factor
    );
    assert_eq!(
        f("moe_router_logit_softcapping") as f32,
        ARTIFACT.moe.router_logit_softcap
    );
    assert_eq!(f("rms_norm_eps") as f32, ARTIFACT.rms_eps);
    assert_eq!(
        c["hidden_act"], "silu",
        "the expert gate transform is SwiGLU"
    );
    assert!(c["norm_topk_prob"].as_bool().unwrap());
    assert!(!c["moe_apply_router_weight_on_input"].as_bool().unwrap());
    assert_eq!(
        c["tie_word_embeddings"].as_bool().unwrap(),
        ARTIFACT.tied_embeddings
    );
    assert_eq!(c["gating"], "per-head");
    assert_eq!(ARTIFACT.attention_gate, AttentionGate::PerHead);

    // The per-layer arrays, against the predicates that stand in for them.
    let types = c["layer_types"].as_array().expect("layer_types");
    let heads = c["num_attention_heads_per_layer"]
        .as_array()
        .expect("num_attention_heads_per_layer");
    let dense = c["mlp_only_layers"].as_array().expect("mlp_only_layers");
    assert_eq!(types.len(), ARTIFACT.layers as usize);
    assert_eq!(heads.len(), ARTIFACT.layers as usize);
    for layer in 0..ARTIFACT.layers {
        let full = types[layer as usize] == "full_attention";
        assert_eq!(
            full,
            ARTIFACT.full_attention_layer(layer),
            "layer {layer} type"
        );
        let want = if full {
            ARTIFACT.full_heads
        } else {
            ARTIFACT.sliding_heads
        };
        assert_eq!(
            heads[layer as usize].as_u64().unwrap(),
            want,
            "layer {layer}"
        );
        assert_eq!(
            dense.iter().any(|d| d.as_u64() == Some(u64::from(layer))),
            ARTIFACT.dense_layer(layer),
            "layer {layer} mlp"
        );
    }

    // The rotary, which is where this family stops being composable.
    let rope = &c["rope_parameters"];
    assert_eq!(rope["full_attention"]["rope_type"], "yarn");
    assert!(matches!(ARTIFACT.full_rope, RopeKind::Yarn));
    assert_eq!(rope["sliding_attention"]["rope_type"], "default");
    let RopeKind::Default {
        theta,
        partial_rotary,
    } = ARTIFACT.sliding_rope
    else {
        panic!("the sliding layers' rotary is the one this workspace can compute");
    };
    assert_eq!(
        rope["sliding_attention"]["rope_theta"].as_f64().unwrap() as f32,
        theta
    );
    assert_eq!(
        rope["sliding_attention"]["partial_rotary_factor"]
            .as_f64()
            .unwrap(),
        partial_rotary.numerator as f64 / partial_rotary.denominator as f64
    );

    // And the quantization: asymmetric INT4 at group 32, which task 0024's
    // importer reads and no kernel executes.
    let q = &c["quantization_config"];
    assert_eq!(q["quant_method"], "compressed-tensors");
    assert_eq!(q["format"], "pack-quantized");
    let w = &q["config_groups"]["group_0"]["weights"];
    assert_eq!(w["num_bits"].as_u64().unwrap(), 4);
    assert_eq!(w["group_size"].as_u64().unwrap(), 32);
    assert!(
        !w["symmetric"].as_bool().unwrap(),
        "asymmetric, which task 0024's importer reads through \
         `ZeroPointSource::PackedAlongOutput`"
    );
}

/// An intermediate width that overflows its own doubling is refused, not a
/// panic.
///
/// `2 * moe_intermediate` used to be evaluated inline before being handed to
/// the checked multiplication helper, so a review reached a debug-build panic
/// through `BlockConfig` -- a public type whose constructor had already
/// accepted the value. Checked arithmetic that runs after the overflow is not
/// checked arithmetic.
#[test]
fn an_overflowing_intermediate_width_is_refused_rather_than_panicking() {
    let mut config = BlockConfig::reduced();
    config.moe.moe_intermediate = 1 << 63;
    let model = RoutedBlocks::reduced(config, "probe").expect("the width is checked at compose");
    let err = model
        .compose(&oracles(), SymbolId(0))
        .expect_err("compose accepted an intermediate that cannot be doubled");
    assert!(format!("{err}").contains("moe_intermediate"), "{err}");

    // Every other product on this path is refused too, and the stage differs by
    // field -- which is worth asserting rather than papering over, because
    // "refused somewhere" is the property and "refused at compose" is not.
    let mut wide_hidden = BlockConfig::reduced();
    wide_hidden.hidden = 1 << 62;
    let model = RoutedBlocks::reduced(wide_hidden, "probe").expect("hidden is checked at compose");
    assert!(model.compose(&oracles(), SymbolId(0)).is_err());

    // A vocabulary that cannot be a `u32` never reaches composition: the
    // metadata refuses it first, because `ModelMetadata::vocab_size` is one.
    let mut wide_vocab = BlockConfig::reduced();
    wide_vocab.vocab = 1 << 62;
    let err = RoutedBlocks::reduced(wide_vocab, "probe").expect_err("vocabulary");
    assert!(format!("{err}").contains("vocabulary"), "{err}");
}
