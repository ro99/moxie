//! Host-only acceptance fixture for task 0040's unabsorbed MLA path.

use moxie_graph::{
    AttentionOutputReduction, Bindings, Graph, GraphBuilder, IndexEncoding, KvHeadPartition,
    MlaAttentionDescriptor, OpParams, OracleRegistry, PartitionRule, TensorSpec, ValueId,
    ValueRole, Visibility,
};
use moxie_interp::{Cancel, HostTensor, Interpreter, KvCache, Value};
use moxie_oracles::mla::{self, MlaCachedToken, MlaWeights};
use moxie_state::{MlaLatentDescriptor, ROOT, SequenceState, StateKind};
use moxie_types::{ActivationPrecision, CachePrecision, Dim, Precision, SymbolId, WeightPrecision};

// Predeclared before any comparison: this is two conservative BF16 units for
// the final hidden row, after widening the oracle's FP64 result to f32 and
// applying the graph's final BF16 boundary.
const OUTPUT_TOLERANCE: f64 = 0.015_625; // 2 * 2^-7

#[derive(Debug, Clone)]
struct FixtureWeights {
    q_a_proj: Vec<f32>,
    q_a_layernorm: Vec<f32>,
    q_b_proj: Vec<f32>,
    kv_a_proj_with_mqa: Vec<f32>,
    kv_a_layernorm: Vec<f32>,
    kv_b_proj: Vec<f32>,
    o_proj: Vec<f32>,
}

impl FixtureWeights {
    fn f64(&self) -> OwnedF64Weights {
        OwnedF64Weights {
            q_a_proj: self.q_a_proj.iter().map(|v| *v as f64).collect(),
            q_a_layernorm: self.q_a_layernorm.iter().map(|v| *v as f64).collect(),
            q_b_proj: self.q_b_proj.iter().map(|v| *v as f64).collect(),
            kv_a_proj_with_mqa: self.kv_a_proj_with_mqa.iter().map(|v| *v as f64).collect(),
            kv_a_layernorm: self.kv_a_layernorm.iter().map(|v| *v as f64).collect(),
            kv_b_proj: self.kv_b_proj.iter().map(|v| *v as f64).collect(),
            o_proj: self.o_proj.iter().map(|v| *v as f64).collect(),
        }
    }
}

struct OwnedF64Weights {
    q_a_proj: Vec<f64>,
    q_a_layernorm: Vec<f64>,
    q_b_proj: Vec<f64>,
    kv_a_proj_with_mqa: Vec<f64>,
    kv_a_layernorm: Vec<f64>,
    kv_b_proj: Vec<f64>,
    o_proj: Vec<f64>,
}

impl OwnedF64Weights {
    fn refs(&self) -> MlaWeights<'_> {
        MlaWeights {
            q_a_proj: &self.q_a_proj,
            q_a_layernorm: &self.q_a_layernorm,
            q_b_proj: &self.q_b_proj,
            kv_a_proj_with_mqa: &self.kv_a_proj_with_mqa,
            kv_a_layernorm: &self.kv_a_layernorm,
            kv_b_proj: &self.kv_b_proj,
            o_proj: &self.o_proj,
        }
    }
}

struct Fixture {
    graph: Graph,
    descriptor: MlaAttentionDescriptor,
    weights: FixtureWeights,
    weight_bindings: Vec<(ValueId, Vec<f32>, Vec<usize>)>,
    input: ValueId,
    positions: ValueId,
}

fn role_activation() -> ValueRole {
    ValueRole::Activation(ActivationPrecision::expect(Precision::Bf16))
}

fn role_weight() -> ValueRole {
    ValueRole::Weight(WeightPrecision::expect(Precision::Bf16))
}

fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let raw = ((i * 11 + seed * 7) % 31) as f32 / 31.0 - 0.5;
            moxie_interp::tensor::to_bf16(raw * 0.5)
        })
        .collect()
}

fn build() -> Fixture {
    let descriptor = MlaAttentionDescriptor {
        hidden: 4,
        q_lora_rank: 2,
        kv_lora_rank: 2,
        qk_nope_head_dim: 2,
        qk_rope_head_dim: 2,
        v_head_dim: 2,
        heads: 2,
        rms_norm_eps: 1e-5,
        rope_base: 10_000.0,
        rope_layout: moxie_graph::RopeLayout::Interleaved,
        visibility: Visibility::Causal,
        layer: 0,
        cache_precision: CachePrecision::expect(Precision::Bf16),
    };
    let mut builder = GraphBuilder::new(moxie_oracles::HOST_REFERENCE, SymbolId(77));
    let rows = Dim::symbol(SymbolId(77));
    let input = builder.input(
        "hidden",
        TensorSpec::new(role_activation(), vec![rows.clone(), Dim::constant(4)]),
    );
    let positions = builder.input(
        "positions",
        TensorSpec::new(ValueRole::Index(IndexEncoding::U64), vec![rows.clone()]),
    );
    let weights = FixtureWeights {
        q_a_proj: values(2 * 4, 1),
        q_a_layernorm: vec![1.0; 2],
        q_b_proj: values(8 * 2, 2),
        kv_a_proj_with_mqa: values(4 * 4, 3),
        kv_a_layernorm: vec![1.0; 2],
        kv_b_proj: values(8 * 2, 4),
        o_proj: values(4 * 4, 5),
    };

    let mut weight_bindings = Vec::new();
    let mut add_matrix = |name: &str, data: Vec<f32>, shape: Vec<usize>| {
        let id = builder
            .weight(
                name,
                TensorSpec::new(
                    role_weight(),
                    shape.iter().map(|d| Dim::constant(*d as u64)).collect(),
                ),
            )
            .unwrap();
        weight_bindings.push((id, data, shape));
        id
    };
    let q_a = add_matrix("q_a_proj", weights.q_a_proj.clone(), vec![2, 4]);
    let q_norm = add_matrix("q_a_layernorm", weights.q_a_layernorm.clone(), vec![2]);
    let q_b = add_matrix("q_b_proj", weights.q_b_proj.clone(), vec![8, 2]);
    let kv_a = add_matrix(
        "kv_a_proj_with_mqa",
        weights.kv_a_proj_with_mqa.clone(),
        vec![4, 4],
    );
    let kv_norm = add_matrix("kv_a_layernorm", weights.kv_a_layernorm.clone(), vec![2]);
    let kv_b = add_matrix("kv_b_proj", weights.kv_b_proj.clone(), vec![8, 2]);
    let o = add_matrix("o_proj", weights.o_proj.clone(), vec![4, 4]);
    let output = builder
        .node(
            OpParams::MlaAttention { descriptor },
            &[input, positions, q_a, q_norm, q_b, kv_a, kv_norm, kv_b, o],
        )
        .unwrap();
    let mut oracles = OracleRegistry::new();
    moxie_oracles::register(&mut oracles).unwrap();
    let graph = builder.finish(output, &oracles).unwrap();
    Fixture {
        graph,
        descriptor,
        weights,
        weight_bindings,
        input,
        positions,
    }
}

fn bindings(fixture: &Fixture, rows: &[f32], positions: &[u64]) -> Bindings<Value> {
    let mut bindings = Bindings::new();
    bindings.set(
        fixture.input,
        Value::Float(HostTensor::bf16(rows.to_vec(), vec![positions.len(), 4]).unwrap()),
    );
    bindings.set(fixture.positions, Value::Index(positions.to_vec()));
    for (id, data, shape) in &fixture.weight_bindings {
        bindings.set(
            *id,
            Value::Float(HostTensor::bf16(data.clone(), shape.clone()).unwrap()),
        );
    }
    bindings
}

fn oracle_outputs(fixture: &Fixture, rows: &[Vec<f32>], positions: &[u64]) -> Vec<Vec<f64>> {
    let weights = fixture.weights.f64();
    let mut history = Vec::new();
    let mut outputs = Vec::new();
    for (row, position) in rows.iter().zip(positions) {
        let input: Vec<f64> = row.iter().map(|v| *v as f64).collect();
        let projection =
            mla::project(fixture.descriptor, &input, weights.refs(), *position).unwrap();
        let token = projection.cached_token().unwrap();
        history.push(MlaCachedToken {
            position: token.position,
            latent: token
                .latent
                .iter()
                .map(|v| moxie_oracles::bf16_round(*v as f32) as f64)
                .collect(),
            rope: token
                .rope
                .iter()
                .map(|v| moxie_oracles::bf16_round(*v as f32) as f64)
                .collect(),
        });
        outputs.push(
            mla::attend(
                fixture.descriptor,
                &projection,
                &history,
                &weights.kv_b_proj,
                &weights.o_proj,
            )
            .unwrap()
            .output,
        );
    }
    outputs
}

fn assert_close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (*got as f64 - *want).abs() <= OUTPUT_TOLERANCE,
            "element {index}: got {got}, expected {want}, tolerance {OUTPUT_TOLERANCE}"
        );
    }
}

#[test]
fn mla_plan_and_interpreter_read_latent_cache_across_prefill_and_decode() {
    let fixture = build();
    assert_eq!(fixture.graph.attention_layers(), vec![0]);
    let node = &fixture.graph.nodes()[0];
    assert_eq!(
        node.contract.partition,
        PartitionRule::HeadShardable {
            kv: KvHeadPartition::SharedLatentReplicated,
            output: AttentionOutputReduction::GlobalReduction,
        }
    );

    let hidden_rows = vec![
        vec![0.25, -0.125, 0.5, -0.25],
        vec![-0.375, 0.25, -0.125, 0.375],
        vec![0.125, 0.5, -0.25, -0.375],
    ];
    let flat: Vec<f32> = hidden_rows.iter().flatten().copied().collect();
    let mut state = SequenceState::new([StateKind::MlaLatent]);
    state.append_prompt(ROOT, 3).unwrap();
    let descriptor =
        MlaLatentDescriptor::new(2, 2, CachePrecision::expect(Precision::Bf16)).unwrap();
    let mut cache = KvCache::for_mla_branch(1, descriptor, &state, ROOT).unwrap();
    let interpreter = Interpreter::new();

    let prefill = interpreter
        .run(
            &fixture.graph,
            &bindings(&fixture, &flat, &[0, 1, 2]),
            &mut state,
            ROOT,
            &mut cache,
            &Cancel::never(),
        )
        .unwrap();
    let expected_prefill = oracle_outputs(&fixture, &hidden_rows, &[0, 1, 2]);
    let expected_prefill_flat: Vec<f64> = expected_prefill.iter().flatten().copied().collect();
    assert_close(prefill.logits.data(), &expected_prefill_flat);
    assert_eq!(cache.mla_contents().unwrap()[0].len(), 3);
    let prefill_rows = cache.mla_contents().unwrap()[0].to_vec();

    let decode_row = vec![vec![0.375, -0.25, 0.125, 0.5]];
    let decode = interpreter
        .run(
            &fixture.graph,
            &bindings(&fixture, &decode_row[0], &[3]),
            &mut state,
            ROOT,
            &mut cache,
            &Cancel::never(),
        )
        .unwrap();
    let mut all_rows = hidden_rows.clone();
    all_rows.extend(decode_row.clone());
    let expected_decode = oracle_outputs(&fixture, &all_rows, &[0, 1, 2, 3]);
    assert_close(decode.logits.data(), &expected_decode[3]);
    assert_eq!(cache.mla_contents().unwrap()[0].len(), 4);

    // A cancellation after publication has started must abort both journals;
    // no tentative latent row may survive.
    let before_cancel = cache.mla_contents().unwrap()[0].to_vec();
    let before_frontier = state.frontiers(ROOT).unwrap();
    let cancelled = interpreter.run(
        &fixture.graph,
        &bindings(&fixture, &[0.125, -0.5, 0.25, -0.125], &[4]),
        &mut state,
        ROOT,
        &mut cache,
        &Cancel::after(3),
    );
    assert!(cancelled.is_err());
    assert_eq!(state.frontiers(ROOT).unwrap(), before_frontier);
    assert_eq!(cache.mla_contents().unwrap()[0], before_cancel);

    // StateKind::MlaLatent advertises truncation, so both physical and logical
    // state can discard the suffix without touching the retained prefix.
    state.rollback_to(ROOT, 3, &[]).unwrap();
    cache.rollback_to(&state, ROOT, 3).unwrap();
    assert_eq!(cache.mla_contents().unwrap()[0], prefill_rows);
    cache.check_owner(&state, ROOT).unwrap();
}

#[test]
fn mla_cache_descriptor_mismatch_is_refused_before_execution() {
    let fixture = build();
    let mut state = SequenceState::new([StateKind::MlaLatent]);
    state.append_prompt(ROOT, 1).unwrap();
    let wrong = MlaLatentDescriptor::new(3, 2, CachePrecision::expect(Precision::Bf16)).unwrap();
    let mut cache = KvCache::for_mla_branch(1, wrong, &state, ROOT).unwrap();
    let error = Interpreter::new()
        .run(
            &fixture.graph,
            &bindings(&fixture, &[0.25, -0.125, 0.5, -0.25], &[0]),
            &mut state,
            ROOT,
            &mut cache,
            &Cancel::never(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), "invalid_request");
    assert!(error.to_string().contains("mla_descriptor"));
}
